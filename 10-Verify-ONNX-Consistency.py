import onnxruntime as ort
import numpy as np
import os
import torch
import torch.nn.functional as F
import torchaudio

def verify_onnx():
    old_path = r'./model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
    new_path = r'./export_onnx_paddable/encoder_paddable.onnx'
    
    print("--- ONNX Consistency Verification ---")
    
    # [0] Inspect Input Names
    sess_old = ort.InferenceSession(old_path, providers=['CPUExecutionProvider'])
    input_name_old = sess_old.get_inputs()[0].name
    print(f"Old Model Input Name: {input_name_old}")
    
    sess_new = ort.InferenceSession(new_path, providers=['CPUExecutionProvider'])
    print(f"New Model Input Names: {[i.name for i in sess_new.get_inputs()]}")
    
    # [1] Data Preparation
    SAMPLE_RATE = 16000
    torch.manual_seed(42)
    s5_samples = SAMPLE_RATE * 5
    audio_5s_torch = torch.randn(1, 1, s5_samples)
    audio_5s_np = audio_5s_torch.numpy()
    
    s30_samples = SAMPLE_RATE * 30
    audio_30s_torch = F.pad(audio_5s_torch, (0, s30_samples - s5_samples), value=0.0)
    audio_30s_np = audio_30s_torch.numpy()
    ilens_5s_np = np.array([s5_samples], dtype=np.int64)

    # [2] Test Round 1: 5s Baseline (Old 5s vs New 5s)
    print("\n[Round 1] Baseline: Old ONNX (5s) vs New ONNX (5s)")
    
    # Run Old
    res_old_5s = sess_old.run(None, {input_name_old: audio_5s_np})
    # Encoder output is usually index 0, or check names
    # Fun-ASR old usually outputs [encoder_out, adaptor_out]
    old_adapt_5s = res_old_5s[1]
    
    # Run New
    res_new_5s = sess_new.run(None, {'audio': audio_5s_np, 'ilens': ilens_5s_np})
    new_adapt_5s = res_new_5s[1]
    
    # Target length logic
    T_mel_valid = (s5_samples // 160) + 1
    T_lfr_valid = (T_mel_valid + 6 - 1) // 6
    olens_1 = 1 + (T_lfr_valid - 3 + 2) // 2
    target_len = (1 + (olens_1 - 3 + 2) // 2 - 1) // 2 + 1
    
    new_adapt_5s_valid = new_adapt_5s[:, :target_len, :]
    
    def compare(t1, t2, label):
        diff = np.abs(t1 - t2)
        max_err = np.max(diff)
        mean_err = np.mean(diff)
        
        # Cosine Similarity using numpy
        # output shape is (1, T, 1024)
        t1_f = t1.reshape(-1, t1.shape[-1])
        t2_f = t2.reshape(-1, t2.shape[-1])
        
        norm1 = np.linalg.norm(t1_f, axis=1, keepdims=True)
        norm2 = np.linalg.norm(t2_f, axis=1, keepdims=True)
        dot_product = np.sum(t1_f * t2_f, axis=1, keepdims=True)
        cos_sim = np.mean(dot_product / (norm1 * norm2 + 1e-12))
        
        print(f"  {label}: MaxErr={max_err:.6e}, MeanErr={mean_err:.6e}, CosSim={cos_sim:.10f}")
        if max_err < 2e-3: print(f"  ✅ {label} Consistent")
        else: print(f"  ❌ {label} Discrepancy detected")

    compare(old_adapt_5s, new_adapt_5s_valid, "Baseline Consistency")

    # [3] Test Round 2: Padding Robustness (Old 5s vs New 30s)
    print("\n[Round 2] Robustness: Old ONNX (5s) vs New ONNX (30s Padded)")
    
    res_new_30s = sess_new.run(None, {'audio': audio_30s_np, 'ilens': ilens_5s_np})
    new_adapt_30s_valid = res_new_30s[1][:, :target_len, :]
    
    compare(old_adapt_5s, new_adapt_30s_valid, "Padding Consistency")

if __name__ == "__main__":
    verify_onnx()
