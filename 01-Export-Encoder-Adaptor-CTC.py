import os
import sys
import warnings
import logging
import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
import onnx
import base64
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# Suppress specific warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# Import the consolidated model definitions
import fun_asr_gguf.model_definition as model_def

# =========================================================================
# Configuration
# =========================================================================

OUTPUT_DIR = r'./model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_dir = r'./Fun-ASR-Nano-2512'
weight_path = os.path.join(model_dir, "model.pt")

onnx_encoder_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
onnx_ctc_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.fp32.onnx'
tokens_path = f'{OUTPUT_DIR}/tokens.txt'

SAMPLE_RATE = 16000
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
OPSET = 18

# =========================================================================
# Utils
# =========================================================================

def generate_sensevoice_vocab(tiktoken_path):
    print(f"Generating vocabulary from {tiktoken_path}...")
    tokens = []
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): tokens.append(line.split()[0])
    
    special_labels = [
        "<|endoftext|>", "<|startoftranscript|>",
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", 
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", 
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", 
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", 
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", 
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", 
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", 
        "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause",
        "HAPPY", "SAD", "ANGRY", "NEUTRAL",
        "translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps"
    ]
    for label in special_labels:
        if not label.startswith("<|"): label = f"<|{label}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
    for i in range(1, 51): tokens.append(base64.b64encode(f"<|SPECIAL_TOKEN_{i}|>".encode()).decode())
    for i in range(1500): tokens.append(base64.b64encode(f"<|{i * 0.02:.2f}|>".encode()).decode())
    tokens.append(base64.b64encode("<blk>".encode()).decode())
    return tokens

def merge_onnx_file(model_path):
    print(f"   [Merge] Merging external data for {os.path.basename(model_path)}...")
    data_file = model_path + ".data"
    try:
        model = onnx.load(model_path)
        onnx.save(model, model_path)
        print(f"   [Merge] Successfully embedded weights.")
        if os.path.exists(data_file):
            os.remove(data_file)
            print(f"   [Cleanup] Deleted external data file.")
    except Exception as e:
        print(f"   [Warning] Merge failed: {e}")

# =========================================================================
# Main Export
# =========================================================================

def main():
    print("\n[Hybrid Export] Consolidated Model Definitions Test...")
    
    tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
    if os.path.exists(tiktoken_path):
        tokens = generate_sensevoice_vocab(tiktoken_path)
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(tokens): f.write(f"{t} {i}\n")
    else:
        print("Warning: tiktoken file not found, vocab generation skipped.")
        tokens = ["dummy"] * 60515 
    
    hybrid = model_def.HybridSenseVoice(vocab_size=len(tokens))
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    custom_stft = model_def.STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)

    with torch.no_grad():
        print(f"\n[1/2] Exporting Encoder-Adaptor (Dynamo=True)...")
        enc_wrapper = model_def.EncoderExportWrapper(hybrid, custom_stft, fbank).eval()
        audio = torch.randn(1, 1, SAMPLE_RATE * 1) 
        
        torch.onnx.export(
            enc_wrapper, (audio,), onnx_encoder_fp32,
            input_names=['audio'], 
            output_names=['enc_output', 'adaptor_output'],
            dynamic_axes={
                'audio': {2: 'audio_len'}, 
                'enc_output': {1: 'enc_len'}, 
                'adaptor_output': {1: 'adaptor_len'}
            },
            opset_version=OPSET,
            dynamo=True
        )
        merge_onnx_file(onnx_encoder_fp32)

        print(f"\n[2/2] Exporting CTC Head (Dynamo=True)...")
        ctc_wrapper = model_def.CTCHeadExportWrapper(hybrid).eval()
        dummy_enc = torch.randn(1, 100, 512)
        torch.onnx.export(
            ctc_wrapper, (dummy_enc,), onnx_ctc_fp32,
            input_names=['enc_output'], output_names=['indices'],
            dynamic_axes={'enc_output': {1: 'enc_len'}, 'indices': {1: 'enc_len'}},
            opset_version=OPSET,
            dynamo=True
        )
        merge_onnx_file(onnx_ctc_fp32)
        
    print("\n[Success] Export complete using consolidated module.")

if __name__ == "__main__":
    main()