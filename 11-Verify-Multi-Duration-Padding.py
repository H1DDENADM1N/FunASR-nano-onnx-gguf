import onnxruntime as ort
import numpy as np
import os
import torch
import torch.nn.functional as F

def verify_multi_padding():
    old_path = r'./model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
    new_path = r'./export_onnx_paddable/encoder_paddable.onnx'
    
    print("--- Multi-Duration Padding Robustness Test (Target: 60s) ---")
    
    sess_old = ort.InferenceSession(old_path, providers=['CPUExecutionProvider'])
    input_name_old = sess_old.get_inputs()[0].name
    
    sess_new = ort.InferenceSession(new_path, providers=['CPUExecutionProvider'])
    
    SAMPLE_RATE = 16000
    TARGET_TOTAL_SEC = 60
    S60_SAMPLES = SAMPLE_RATE * TARGET_TOTAL_SEC
    
    durations = [5, 10, 12, 20]
    
    def get_cos_sim(t1, t2):
        t1_f = t1.reshape(-1, t1.shape[-1])
        t2_f = t2.reshape(-1, t2.shape[-1])
        norm1 = np.linalg.norm(t1_f, axis=1, keepdims=True)
        norm2 = np.linalg.norm(t2_f, axis=1, keepdims=True)
        dot_product = np.sum(t1_f * t2_f, axis=1, keepdims=True)
        return np.mean(dot_product / (norm1 * norm2 + 1e-12))

    for sec in durations:
        print(f"\n[Testing {sec}s Audio -> 60s Container]")
        
        # 1. Prepare Data
        torch.manual_seed(42 + sec) # Different seed per duration for variety
        valid_samples = SAMPLE_RATE * sec
        audio_torch = torch.randn(1, 1, valid_samples)
        
        # Original Inference (Old Model, native length)
        res_old = sess_old.run(None, {input_name_old: audio_torch.numpy()})
        old_adapt = res_old[1]
        
        # Padded Inference (New Model, 60s container)
        audio_60s_torch = F.pad(audio_torch, (0, S60_SAMPLES - valid_samples), value=0.0)
        ilens_np = np.array([valid_samples], dtype=np.int64)
        
        res_new = sess_new.run(None, {'audio': audio_60s_torch.numpy(), 'ilens': ilens_np})
        new_adapt_full = res_new[1]
        
        # Calculate target length logic (SenseVoice specific)
        T_mel_valid = (valid_samples // 160) + 1
        T_lfr_valid = (T_mel_valid + 6 - 1) // 6
        olens_1 = 1 + (T_lfr_valid - 3 + 2) // 2
        target_len = (1 + (olens_1 - 3 + 2) // 2 - 1) // 2 + 1
        
        new_adapt_valid = new_adapt_full[:, :target_len, :]
        
        # 2. Compare
        diff = np.abs(old_adapt - new_adapt_valid)
        max_err = np.max(diff)
        sim = get_cos_sim(old_adapt, new_adapt_valid)
        
        print(f"  Result for {sec}s:")
        print(f"    Max Absolute Error: {max_err:.6e}")
        print(f"    Cosine Similarity:  {sim:.12f}")
        
        if sim > 0.999999:
            print(f"  ✅ {sec}s Consistent")
        else:
            print(f"  ❌ {sec}s Discrepancy detected!")

if __name__ == "__main__":
    verify_multi_padding()
