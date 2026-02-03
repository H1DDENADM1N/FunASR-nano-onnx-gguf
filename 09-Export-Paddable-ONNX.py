import torch
import torch.nn.functional as F
import torchaudio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
import fun_asr_gguf.model_definition_paddable as model_paddable

def export_paddable_onnx():
    # --- Configuration ---
    model_dir = r'./Fun-ASR-Nano-2512'
    weight_path = os.path.join(model_dir, "model.pt")
    output_dir = r'./export_onnx_paddable'
    os.makedirs(output_dir, exist_ok=True)
    
    export_path = os.path.join(output_dir, "encoder_paddable.onnx")
    
    SAMPLE_RATE = 16000
    N_MELS = 80
    NFFT_STFT = 400
    WINDOW_LENGTH = 400
    HOP_LENGTH = 160
    
    print("--- Exporting Paddable SenseVoice Encoder to ONNX ---")

    # 1. Load Model
    print("[1] Loading PyTorch model...")
    hybrid = model_paddable.HybridSenseVoice(vocab_size=60515)
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    stft = model_paddable.STFT_Process(n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH).eval()
    fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
    
    wrapper = model_paddable.EncoderExportWrapperPaddable(hybrid, stft, fbank).eval()

    # 2. Prepare Dummy Inputs
    # We use a 5s dummy input for initial trace
    dummy_samples = SAMPLE_RATE * 5
    dummy_audio = torch.randn(1, 1, dummy_samples)
    dummy_ilens = torch.tensor([dummy_samples], dtype=torch.long)

    # 3. Export to ONNX
    print(f"[2] Exporting to {export_path}...")
    
    # We enable dynamic axes for 'samples' so it can handle 5s, 30s, etc.
    torch.onnx.export(
        wrapper,
        (dummy_audio, dummy_ilens),
        export_path,
        input_names=['audio', 'ilens'],
        output_names=['enc_output', 'adapt_output'],
        dynamic_axes={
            'audio': {2: 'samples'},
            'ilens': {0: 'batch'},
            'enc_output': {1: 'enc_frames'},
            'adapt_output': {1: 'adapt_frames'}
        },
        opset_version=18,
        do_constant_folding=True
    )

    print("\n✅ Export Complete!")
    print(f"Model saved to: {os.path.abspath(export_path)}")
    print("\nNext steps:")
    print("1. Use ONNX Runtime to verify this model with DirectML.")
    print("2. Confirm that padding to 30s results in the same output as 5s.")

if __name__ == "__main__":
    export_paddable_onnx()
