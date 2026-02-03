import torch
import numpy as np
import onnxruntime
import time
import os
import sys
import torchaudio

# Add project root to path
sys.path.append(os.getcwd())
import fun_asr_gguf.model_definition as model_def

def verify_consistency():
    # --- Configuration ---
    model_dir = r'./Fun-ASR-Nano-2512'
    weight_path = os.path.join(model_dir, "model.pt")
    onnx_path = r'./model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
    
    SAMPLE_RATE = 16000
    N_MELS = 80
    NFFT_STFT = 400
    WINDOW_LENGTH = 400
    HOP_LENGTH = 160
    LFR_M = 7
    LFR_N = 6
    
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}")
        return

    print("--- Testing Consistency: PyTorch vs. ONNX ---")

    # 1. Load PyTorch Model
    print("[1] Loading PyTorch model...")
    hybrid = model_def.HybridSenseVoice(vocab_size=60515) # Default size
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    custom_stft = model_def.STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
    
    pt_wrapper = model_def.EncoderExportWrapper(hybrid, custom_stft, fbank).eval()

    # 2. Load ONNX Model
    print("[2] Loading ONNX model...")
    ort_sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 3. Prepare Random Input (5 seconds)
    print("[3] Preparing random 5s audio...")
    audio_np = np.random.randn(1, 1, SAMPLE_RATE * 5).astype(np.float32)
    audio_torch = torch.from_numpy(audio_np)

    # 4. Run PyTorch Inference
    print("[4] Running PyTorch inference...")
    with torch.no_grad():
        pt_enc, pt_adapt = pt_wrapper(audio_torch)
    
    # 5. Run ONNX Inference
    print("[5] Running ONNX inference...")
    ort_inputs = {ort_sess.get_inputs()[0].name: audio_np}
    ort_outputs = ort_sess.run(None, ort_inputs)
    ort_enc, ort_adapt = ort_outputs

    # 6. Compare Results
    print("\n--- Results Comparison ---")
    
    def compare(pt_val, ort_val, name):
        pt_val = pt_val.cpu().numpy()
        abs_diff = np.abs(pt_val - ort_val)
        max_err = np.max(abs_diff)
        mean_err = np.mean(abs_diff)
        print(f"  [{name}]")
        print(f"    Shape: pt={pt_val.shape}, ort={ort_val.shape}")
        print(f"    Max Error:  {max_err:.6e}")
        print(f"    Mean Error: {mean_err:.6e}")
        
        if np.allclose(pt_val, ort_val, atol=1e-4):
            print(f"    ✅ {name} matches perfectly (atol=1e-4)")
        else:
            print(f"    ❌ {name} mismatch detected!")

    compare(pt_enc, ort_enc, "Encoder Output")
    compare(pt_adapt, ort_adapt, "Adaptor Output")

if __name__ == "__main__":
    verify_consistency()
