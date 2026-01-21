"""
端到端 ASR 推理脚本 (End-to-End ASR Inference)

流程:
1. 加载音频文件 (MP3/WAV/etc) -> PCM 16kHz
2. ONNX Encoder: 音频 -> Audio Embedding
3. 从 GGUF 读取 token_embd.weight，生成 prefix/suffix embedding
4. 拼接 [prefix + audio + suffix]
5. llama.dll 直接注入 embedding 并生成文本

依赖:
- onnxruntime
- pydub (音频处理)
- gguf (读取 GGUF 模型权重)
- llama.dll / ggml.dll (推理)
"""

import sys
import os
import ctypes
import numpy as np
import time
import gguf

# =========================================================================
# TOGGLE FIX HERE
# =========================================================================
# os.environ["VK_ICD_FILENAMES"] = "none"       # 禁止 Vulkan
# os.environ["GGML_VK_VISIBLE_DEVICES"] = "0"   # 禁止独显
# os.environ["GGML_VK_DISABLE_F16"] = "1"       # 禁止 FP16 计算

# =========================================================================



# =========================================================================
# 配置区 (硬编码，直接修改这些值)
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径
MODEL_DIR = os.path.join(SCRIPT_DIR, "model-gguf")
BIN_DIR = os.path.join(SCRIPT_DIR, "bin")

# GGUF 解码器 - 请确保此路径正确！
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "qwen3-0.6b-asr.gguf")
GGUF_MODEL_PATH = os.path.join(MODEL_DIR, "qwen3-0.6b-asr-q8_0.gguf")

ONNX_ENCODER_PATH = os.path.join(MODEL_DIR, "FunASR_Nano_Encoder.onnx")  # ONNX 编码器

GGML_DLL_PATH = os.path.join(BIN_DIR, "ggml.dll")
LLAMA_DLL_PATH = os.path.join(BIN_DIR, "llama.dll")
GGML_BASE_DLL_PATH = os.path.join(BIN_DIR, "ggml-base.dll")

# 输入音频
INPUT_AUDIO = os.path.join(SCRIPT_DIR, "input.mp3")

# ASR Prompts
PREFIX_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|startofspeech|>"
PREFIX_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n语音转写：<|startofspeech|>"
SUFFIX_PROMPT = "<|endofspeech|>\n<|im_end|>\n<|im_start|>assistant"
STOP_TOKENS = [151643, 151645]

# 音频参数
SAMPLE_RATE = 16000
USE_NORMALIZER = True
MAX_INPUT_AUDIO_LENGTH = 320000  # 最大音频长度 (samples)

# 推理参数
N_PREDICT = 500                  # 最大生成 token 数
N_THREADS = 0                    # 线程数 (0=自动)
QUIET_MODE = True                # 静默模式 (关闭 llama.cpp 调试信息)
INJECT_CHUNK_SIZE = 512          # 注入 Embedding 时的分块大小 
N_UBATCH = 512                   # llama.cpp 内部物理 batch 大小

# =========================================================================
# DLL Loading
# =========================================================================



# Load backends - 临时切换到 bin 目录，让 Windows 找到依赖 DLL (如 MKL, SYCL, Vulkan 等)
original_cwd = os.getcwd()      # 保存原工作目录
os.chdir(BIN_DIR)               # 临时切换到 bin 目录
try:
    ggml = ctypes.CDLL(GGML_DLL_PATH)
    ggml_base = ctypes.CDLL(GGML_BASE_DLL_PATH)
    llama = ctypes.CDLL(LLAMA_DLL_PATH)

    ggml_backend_load_all = ggml.ggml_backend_load_all

    ggml_backend_load_all.argtypes = []
    ggml_backend_load_all.restype = None
    ggml_backend_load_all()         # 加载后端

finally:
    os.chdir(original_cwd)          # 立即切换回原目录

# GGML Device Enumeration Bindings
ggml_backend_dev_count = ggml.ggml_backend_dev_count
ggml_backend_dev_count.argtypes = []
ggml_backend_dev_count.restype = ctypes.c_size_t

ggml_backend_dev_get = ggml.ggml_backend_dev_get
ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
ggml_backend_dev_get.restype = ctypes.c_void_p

ggml_backend_dev_description = ggml_base.ggml_backend_dev_description
ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
ggml_backend_dev_description.restype = ctypes.c_char_p

ggml_backend_dev_name = ggml_base.ggml_backend_dev_name
ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
ggml_backend_dev_name.restype = ctypes.c_char_p


# =========================================================================
# Type Definitions
# =========================================================================

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.POINTER(ctypes.c_void_p)),
        ("tensor_buft_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.POINTER(ctypes.c_float)),
        ("progress_callback", ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.c_void_p)),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.POINTER(ctypes.c_void_p)),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_direct_io", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
    ]

class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(ctypes.c_void_p)),
        ("n_samplers", ctypes.c_size_t),
    ]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens", ctypes.c_int32),
        ("token", ctypes.POINTER(llama_token)),
        ("embd", ctypes.POINTER(ctypes.c_float)),
        ("pos", ctypes.POINTER(llama_pos)),
        ("n_seq_id", ctypes.POINTER(ctypes.c_int32)),
        ("seq_id", ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits", ctypes.POINTER(ctypes.c_int8)),
    ]

# =========================================================================
# Function Prototypes
# =========================================================================

# Logging
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def quiet_log_callback(level, message, user_data):
    pass
llama_log_set = llama.llama_log_set
llama_log_set.argtypes = [LOG_CALLBACK, ctypes.c_void_p]
llama_log_set.restype = None

# Backend
llama_backend_init = llama.llama_backend_init
llama_backend_init.argtypes = []
llama_backend_init.restype = None

llama_backend_free = llama.llama_backend_free
llama_backend_free.argtypes = []
llama_backend_free.restype = None

# Model
llama_model_default_params = llama.llama_model_default_params
llama_model_default_params.argtypes = []
llama_model_default_params.restype = llama_model_params

llama_model_load_from_file = llama.llama_model_load_from_file
llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
llama_model_load_from_file.restype = ctypes.c_void_p

llama_model_free = llama.llama_model_free
llama_model_free.argtypes = [ctypes.c_void_p]
llama_model_free.restype = None

llama_model_get_vocab = llama.llama_model_get_vocab
llama_model_get_vocab.argtypes = [ctypes.c_void_p]
llama_model_get_vocab.restype = ctypes.c_void_p

# Context
llama_context_default_params = llama.llama_context_default_params
llama_context_default_params.argtypes = []
llama_context_default_params.restype = llama_context_params

llama_init_from_model = llama.llama_init_from_model
llama_init_from_model.argtypes = [ctypes.c_void_p, llama_context_params]
llama_init_from_model.restype = ctypes.c_void_p

llama_free = llama.llama_free
llama_free.argtypes = [ctypes.c_void_p]
llama_free.restype = None

# Batch
llama_batch_init = llama.llama_batch_init
llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
llama_batch_init.restype = llama_batch

llama_batch_free = llama.llama_batch_free
llama_batch_free.argtypes = [llama_batch]
llama_batch_free.restype = None

# State
llama_state_get_size = llama.llama_state_get_size
llama_state_get_size.argtypes = [ctypes.c_void_p]
llama_state_get_size.restype = ctypes.c_size_t

llama_state_get_data = llama.llama_state_get_data
llama_state_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
llama_state_get_data.restype = ctypes.c_size_t

llama_state_set_data = llama.llama_state_set_data
llama_state_set_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
llama_state_set_data.restype = ctypes.c_size_t

# Decode
llama_decode = llama.llama_decode
llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
llama_decode.restype = ctypes.c_int32

# Logits
llama_get_logits = llama.llama_get_logits
llama_get_logits.argtypes = [ctypes.c_void_p]
llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)

# Tokenize
llama_tokenize = llama.llama_tokenize
llama_tokenize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(llama_token), ctypes.c_int32,
    ctypes.c_bool, ctypes.c_bool,
]
llama_tokenize.restype = ctypes.c_int32

# Vocab
llama_vocab_n_tokens = llama.llama_vocab_n_tokens
llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
llama_vocab_n_tokens.restype = ctypes.c_int32

llama_vocab_eos = llama.llama_vocab_eos
llama_vocab_eos.argtypes = [ctypes.c_void_p]
llama_vocab_eos.restype = llama_token

llama_token_to_piece = llama.llama_token_to_piece
llama_token_to_piece.argtypes = [ctypes.c_void_p, llama_token, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool]
llama_token_to_piece.restype = ctypes.c_int

# =========================================================================
# Helper Functions
# =========================================================================

def text_to_tokens(vocab, text):
    """Tokenize text using llama.dll"""
    text_bytes = text.encode("utf-8")
    n_tokens_max = len(text_bytes) + 32
    tokens = (llama_token * n_tokens_max)()
    
    n = llama_tokenize(vocab, text_bytes, len(text_bytes), tokens, n_tokens_max, False, True)
    if n < 0:
        return []
    return [tokens[i] for i in range(n)]

def get_token_embeddings_gguf(model_path, cache_dir=None):
    """
    Read token_embd.weight from GGUF using gguf library.
    支持 F16/F32 和 Q8_0 量化格式
    使用缓存机制：首次读取后保存为 .npy 文件，后续直接加载缓存
    """
    # 生成缓存文件路径
    if cache_dir is None:
        cache_dir = os.path.dirname(model_path)
    
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    cache_path = os.path.join(cache_dir, f"{model_name}_token_embd.npy")
    
    # 如果缓存存在且比模型新，直接加载
    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(model_path):
            return np.load(cache_path)
    
    # 从 GGUF 读取
    reader = gguf.GGUFReader(model_path, mode='r')
    
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            # GGML_TYPE_Q8_0 = 8
            if t.tensor_type == 8:
                # Q8_0 解量化
                # Block 结构: d (float16, 2字节) + qs (int8[32], 32字节) = 34 字节
                block_size_bytes = 34
                num_values_per_block = 32
                
                raw_data = t.data
                data_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                n_blocks = data_u8.size // block_size_bytes
                
                blocks = data_u8.reshape(n_blocks, block_size_bytes)
                deltas = blocks[:, :2].view(np.float16).flatten()
                quants = blocks[:, 2:].view(np.int8)
                
                # value = delta * quant
                data = (deltas[:, np.newaxis] * quants).flatten().astype(np.float32).reshape(-1, 1024)
            else:
                # F16 或 F32
                data = t.data
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
            
            # 保存缓存
            np.save(cache_path, data)
            return data
    
    return None

def token_to_bytes(vocab, token_id):
    """Convert token to raw bytes (for BPE byte-level tokens)"""
    buf = ctypes.create_string_buffer(256)
    n = llama_token_to_piece(vocab, token_id, buf, ctypes.sizeof(buf), 0, True)
    if n > 0:
        return buf.raw[:n]
    return b""

class ByteDecoder:
    """
    字节级解码器，用于处理 BPE 拆分的 UTF-8 字符
    累积字节，只有当凑齐完整的 UTF-8 字符时才输出
    """
    def __init__(self):
        self.buffer = b""
    
    def decode(self, raw_bytes):
        self.buffer += raw_bytes
        result = ""
        while self.buffer:
            try:
                result += self.buffer.decode('utf-8')
                self.buffer = b""
                break
            except UnicodeDecodeError as e:
                if e.reason == 'unexpected end of data' or 'invalid continuation' in e.reason:
                    if e.start > 0:
                        result += self.buffer[:e.start].decode('utf-8', errors='replace')
                        self.buffer = self.buffer[e.start:]
                    break
                else:
                    result += self.buffer[:1].decode('utf-8', errors='replace')
                    self.buffer = self.buffer[1:]
        return result
    
    def flush(self):
        if self.buffer:
            result = self.buffer.decode('utf-8', errors='replace')
            self.buffer = b""
            return result
        return ""

def normalizer(audio, target_value=8192.0):
    """音频归一化处理"""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean((audio * audio), dtype=np.float32), dtype=np.float32)
    audio *= (target_value / (rms + 1e-7))
    np.clip(audio, -32768.0, 32767.0, out=audio)
    return audio.astype(np.int16)

def load_audio(audio_path):
    """加载音频文件并转换为 16kHz PCM"""
    from pydub import AudioSegment
    
    audio = np.array(
        AudioSegment.from_file(audio_path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .get_array_of_samples(),
        dtype=np.int16
    )
    
    if USE_NORMALIZER:
        audio = normalizer(audio, 8192.0)
    
    return audio

def encode_audio(audio, ort_session, query_embed):
    """
    使用 ONNX Encoder 将音频转换为 embedding
    
    支持任意长度音频（不做截断，由 ONNX 动态轴处理）
    """
    import onnxruntime
    
    # Reshape: (1, 1, audio_len) - 使用实际长度
    audio_input = audio.reshape(1, 1, -1)
    
    in_names = [x.name for x in ort_session.get_inputs()]
    out_names = [x.name for x in ort_session.get_outputs()]
    
    input_feed = {
        in_names[0]: onnxruntime.OrtValue.ortvalue_from_numpy(audio_input, 'cpu', 0),
        in_names[1]: onnxruntime.OrtValue.ortvalue_from_numpy(query_embed, 'cpu', 0),
    }
    
    outputs = ort_session.run_with_ort_values(out_names, input_feed)
    
    # audio_features: (1, Seq_Len, 1024)
    audio_features = outputs[0].numpy()
    
    # Squeeze to (Seq_Len, 1024)
    return audio_features.squeeze(0)

# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("端到端 ASR 推理 (End-to-End ASR Inference)")
    print("=" * 70)
    
    # Suppress logs if requested
    if QUIET_MODE:
        cb = LOG_CALLBACK(quiet_log_callback)
        llama_log_set(cb, None)
    
    t_start = time.perf_counter()
    
    # =========================================================================
    # 1. 加载 ONNX Audio Encoder
    # =========================================================================
    print("\n[1] 加载 ONNX Audio Encoder...")
    import onnxruntime
    
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # if N_THREADS > 0:
    #     session_opts.intra_op_num_threads = N_THREADS
    #     session_opts.inter_op_num_threads = N_THREADS
    
    model_e_t = time.perf_counter()
    ort_session = onnxruntime.InferenceSession(
        ONNX_ENCODER_PATH, 
        sess_options=session_opts, 
        providers=['CPUExecutionProvider']
    )
    model_e_time = time.perf_counter() - model_e_t

    
    query_embed = np.ones((1, 10, 1024), dtype=np.float32)
    print(f"    Encoder: {os.path.basename(ONNX_ENCODER_PATH)}")
    
    # =========================================================================
    # 2. 加载 GGUF 解码器模型
    # =========================================================================
    print(f"\n[2] 加载 GGUF LLM Decoder")
    model_d_t = time.perf_counter()
    
    llama_backend_init()
    
    model_params = llama_model_default_params()
    # model_params.n_gpu_layers = N_GPU_LAYERS
    # model_params.main_gpu = MAIN_GPU      # 指定使用哪个 GPU
    # model_params.split_mode = SPLIT_MODE  # 0=不分割，全部放在 main_gpu 上
    
    # # Custom Device Selection (Force Intel Arc / Integrated GPU)
    # custom_devices_array = None # Keep reference alive
    
    # if USE_CUSTOM_DEVICE_SELECTION:
    #     print("    [Custom Device Selection] Scanning for Intel/Integrated GPU...")
    #     dev_count = ggml_backend_dev_count()
    #     target_dev = None
        
    #     for i in range(dev_count):
    #         dev = ggml_backend_dev_get(i)
    #         desc_b = ggml_backend_dev_description(dev)
    #         name_b = ggml_backend_dev_name(dev)
            
    #         desc = desc_b.decode('utf-8') if desc_b else "Unknown"
    #         name = name_b.decode('utf-8') if name_b else "Unknown"
            
    #         print(f"      Device {i}: {name} - {desc}")
            
    #         if "Intel" in desc and "Arc" in desc: # Prefer Arc if available
    #              target_dev = dev
    #         elif "Intel" in desc and target_dev is None: # Fallback to any Intel
    #              target_dev = dev
        
    #     if target_dev:
    #         print(f"    [Custom Device] Selected: {ggml_backend_dev_description(target_dev).decode('utf-8')}")
            
    #         # Create a null-terminated array of device pointers
    #         # ggml_backend_dev_t * devices;
    #         DevicesArray = ctypes.c_void_p * 2
    #         custom_devices_array = DevicesArray(target_dev, None)
    #         model_params.devices = ctypes.cast(custom_devices_array, ctypes.POINTER(ctypes.c_void_p))
    #         model_params.main_gpu = 0 # Since we passed a single device list, it becomes index 0
    #     else:
    #         print("    [Custom Device] Warning: No Intel device found. Falling back to default.")

    model = llama_model_load_from_file(GGUF_MODEL_PATH.encode('utf-8'), model_params)
    if not model:
        print("    ERROR: Failed to load model")
        return 1
    
    model_d_time = time.perf_counter() - model_d_t
    print(f"    Decoder: {os.path.basename(GGUF_MODEL_PATH)} (耗时: {model_d_time:.2f}s)")
    
    vocab = llama_model_get_vocab(model)
    vocab_size = llama_vocab_n_tokens(vocab)
    eos_token = llama_vocab_eos(vocab)
    
    # =========================================================================
    # 3. 读取 token embedding 权重
    # =========================================================================
    print("\n[3] 读取 token embedding 权重...")
    embd_read_t = time.perf_counter()
    
    embedding_table = get_token_embeddings_gguf(GGUF_MODEL_PATH)
    if embedding_table is None:
        print("    ERROR: Failed to read token_embd.weight")
        llama_model_free(model)
        return 1
    
    n_vocab, n_embd = embedding_table.shape
    embd_read_time = time.perf_counter() - embd_read_t
    print(f"    Embedding table: {embedding_table.shape} (耗时: {embd_read_time*1000:.2f}ms)")
    
    # =========================================================================
    # 4. 生成 prefix/suffix embeddings
    # =========================================================================
    print("\n[4] 生成 prefix/suffix embeddings...")
    
    prefix_tokens = text_to_tokens(vocab, PREFIX_PROMPT)
    suffix_tokens = text_to_tokens(vocab, SUFFIX_PROMPT)
    
    
    prefix_embd = embedding_table[prefix_tokens].astype(np.float32)
    suffix_embd = embedding_table[suffix_tokens].astype(np.float32)
    
    print(f"    Prefix: {len(prefix_tokens)} tokens -> {prefix_embd.shape}")
    print(f"    Suffix: {len(suffix_tokens)} tokens -> {suffix_embd.shape}")
    
    print("\n" + "=" * 70)
    print("模型加载完成，准备处理音频...")
    print("=" * 70)
    
    # =========================================================================
    # 5. 加载并编码音频
    # =========================================================================
    print(f"\n[5] 加载并编码音频: {os.path.basename(INPUT_AUDIO)}")
    # input("按回车键继续...")
    
    audio = load_audio(INPUT_AUDIO)
    audio_len = len(audio)
    print(f"    音频长度: {audio_len} samples ({audio_len/SAMPLE_RATE:.2f}s)")
    
    t_encode = time.perf_counter()
    audio_embd = encode_audio(audio, ort_session, query_embed)
    encode_time = time.perf_counter() - t_encode
    
    print(f"    Audio Embedding: {audio_embd.shape} (耗时: {encode_time*1000:.2f}ms)")
    print(f"    Audio Embedding: {audio_embd.shape} (耗时: {encode_time*1000:.2f}ms)")
    
    def analyze_tensor(name, tensor):
        return f"{name}: Shape={tensor.shape}, Min={tensor.min():.4f}, Max={tensor.max():.4f}, Mean={tensor.mean():.4f}, Std={tensor.std():.4f}"

    print(analyze_tensor("    Prefix Embd", prefix_embd))
    print(analyze_tensor("    Audio  Embd", audio_embd))
    print(analyze_tensor("    Suffix Embd", suffix_embd))
    
    # Check for NaNs
    if np.isnan(audio_embd).any() or np.isinf(audio_embd).any():
        print("    [WARNING] Audio embedding contains NaN or Inf!")
    if np.isnan(audio_embd).any() or np.isinf(audio_embd).any():
        print("    [WARNING] Audio embedding contains NaN or Inf!")

    
    # =========================================================================
    # 6. 拼接 embeddings
    # =========================================================================
    print("\n[6] 拼接 embeddings [prefix + audio + suffix]...")
    
    audio_embd = audio_embd.astype(np.float32)
    full_embd = np.concatenate([prefix_embd, audio_embd, suffix_embd], axis=0)
    full_embd = np.ascontiguousarray(full_embd)
    n_tokens_input = full_embd.shape[0]
    
    print(f"    总 embedding: {full_embd.shape}")
    
    # =========================================================================
    # 7. 创建上下文并注入 embeddings
    # =========================================================================
    print(f"\n[7] 注入 embeddings ({n_tokens_input} tokens)...")
    
    ctx_params = llama_context_default_params()
    ctx_params.n_ctx = n_tokens_input + N_PREDICT + 64
    ctx_params.n_batch = n_tokens_input
    ctx_params.n_ubatch = N_UBATCH
    ctx_params.embeddings = False  # 关闭 Output Embeddings 计算 (只做 Generation，不需要提取 embedding)
    ctx_params.no_perf = True
    if N_THREADS > 0:
        ctx_params.n_threads = N_THREADS
        ctx_params.n_threads_batch = N_THREADS
    
    ctx = llama_init_from_model(model, ctx_params)
    if not ctx:
        print("    ERROR: Failed to create context")
        llama_model_free(model)
        return 1
    
    batch_embd = llama_batch_init(n_tokens_input, n_embd, 1)
    
    # 循环分块注入 (即便 chunk size 很大，循环逻辑也能保证通用性)
    
    total_chunks = (n_tokens_input + INJECT_CHUNK_SIZE - 1) // INJECT_CHUNK_SIZE
    print(f"    分块注入: Total={n_tokens_input}, Chunk={INJECT_CHUNK_SIZE}, Steps={total_chunks}")
    
    t_inject = time.perf_counter()
    
    for i in range(0, n_tokens_input, INJECT_CHUNK_SIZE):
        end = min(i + INJECT_CHUNK_SIZE, n_tokens_input)
        chunk_len = end - i
        
        # 准备 batch 数据
        batch_embd.n_tokens = chunk_len
        # 注意: token 指针设为 nullptr 表示使用 embd
        batch_embd.token = ctypes.cast(None, ctypes.POINTER(llama_token))
        
        # 提取当前 chunk 的 embedding 并拷贝到 batch
        chunk_data = full_embd[i:end]
        if not chunk_data.flags['C_CONTIGUOUS']:
            chunk_data = np.ascontiguousarray(chunk_data)
        
        ctypes.memmove(batch_embd.embd, chunk_data.ctypes.data, chunk_data.nbytes)
        
        # 设置 pos 和 seq_id
        for k in range(chunk_len):
            batch_embd.pos[k] = i + k
            batch_embd.n_seq_id[k] = 1
            batch_embd.seq_id[k][0] = 0
            # 仅在最后一个 token 开启 logits (其实这里全都可以关掉，因为是 prefill)
            # 但为了逻辑一致，保留最后一个 token 的 logits 标记
            is_global_last = (i + k == n_tokens_input - 1)
            batch_embd.logits[k] = 1 if is_global_last else 0
            
        # 解码
        ret = llama_decode(ctx, batch_embd)
        if ret != 0:
            print(f"    ERROR: Decode failed at chunk {i}-{end} (ret={ret})")
            llama_batch_free(batch_embd)
            llama_free(ctx)
            llama_model_free(model)
            return 1
            
        # 简单进度显示
        # print(".", end="", flush=True)

    inject_time = time.perf_counter() - t_inject
    print() # Newline after progress
    
    print(f"    注入耗时: {inject_time*1000:.2f}ms")
    llama_batch_free(batch_embd)
    
    # =========================================================================
    # 8. 生成文本
    # =========================================================================
    print(f"\n[8] 生成文本 (最大 {N_PREDICT} tokens)...")
    print("=" * 70)
    
    batch_text = llama_batch_init(1, 0, 1)
    batch_text.n_tokens = 1
    
    generated_text = ""
    current_pos = n_tokens_input
    tokens_generated = 0
    decoder = ByteDecoder()
    
    t_gen = time.perf_counter()
    
    try:
        for _ in range(N_PREDICT):
            logits_ptr = llama_get_logits(ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))
            
            if token_id == eos_token or token_id in STOP_TOKENS:
                break
            
            raw_bytes = token_to_bytes(vocab, token_id)
            text_piece = decoder.decode(raw_bytes)
            
            print(text_piece, end="", flush=True)
            generated_text += text_piece
            tokens_generated += 1
            
            batch_text.token[0] = token_id
            batch_text.pos[0] = current_pos
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            
            current_pos += 1
            
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    
    remaining = decoder.flush()
    if remaining:
        print(remaining, end="", flush=True)
        generated_text += remaining
    
    gen_time = time.perf_counter() - t_gen
    
    print()
    print("=" * 70)
    
    # =========================================================================
    # 9. 统计信息
    # =========================================================================
    tps = tokens_generated / gen_time if gen_time > 0 else 0
    total_time = time.perf_counter() - t_start
    
    print(f"\n[结果]")
    print(f"  转录文本: {generated_text}")
    print(f"\n[统计]")
    print(f"  音频长度: {audio_len/SAMPLE_RATE:.2f}s")
    print(f"  Decoder输入: {n_tokens_input} (prefix:{len(prefix_tokens)}, audio:{audio_embd.shape[0]}, suffix:{len(suffix_tokens)})")
    print(f"  Decoder输出: {tps:.2f} tokens/s ({tokens_generated} in {gen_time:.2f}s)")
    print(f"\n[耗时]")
    print(f"  - Encoder加载:   {model_e_time*1000:5.0f}ms")
    print(f"  - Decoder加载:   {model_d_time*1000:5.0f}ms")
    print(f"  - Embd 读取:     {embd_read_time*1000:5.0f}ms")
    print(f"  - 音频编码:      {encode_time*1000:5.0f}ms")
    print(f"  - Decoder读取:   {inject_time*1000:5.0f}ms")
    print(f"  - Decoder生成:   {gen_time*1000:5.0f}ms")
    print(f"  - 转录总耗时:        {encode_time + inject_time + gen_time:5.2f}s")
    
    # Cleanup
    llama_batch_free(batch_text)
    llama_free(ctx)
    llama_model_free(model)
    llama_backend_free()
    
    print("\n[完成]")
    return 0

if __name__ == "__main__":
    exit(main())

