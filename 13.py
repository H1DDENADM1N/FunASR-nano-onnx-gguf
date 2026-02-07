import os

import onnx
import torch
import torchaudio
from onnxruntime.transformers.float16 import convert_float_to_float16

import fun_asr_gguf.model_definition_debug as model_def

# Configuration
OUTPUT_DIR = "./model_debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
weight_path = "./Fun-ASR-Nano-2512/model.pt"
onnx_fp32_path = os.path.join(OUTPUT_DIR, "debug_fp32.onnx")
onnx_fp16_path = os.path.join(OUTPUT_DIR, "debug_fp16.onnx")

SAMPLE_RATE = 16000
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
OPSET = 18


def main():
    """导出FunASR模型的调试版本ONNX格式，包含完整中间层输出并支持FP16转换。

    该函数执行以下主要操作：
    1. 导出FP32精度的ONNX模型，包含所有中间层输出用于调试
    2. 将FP32模型转换为FP16精度（保留LayerNormalization层为FP32）
    3. 保存两种精度的模型文件到指定输出目录

    函数无参数，无返回值。所有配置通过全局常量定义。
    """
    print("--- Exporting Debug FP32 ONNX (Full Layers) ---")
    # 初始化HybridSenseVoice调试模型并加载预训练权重
    hybrid = model_def.HybridSenseVoiceDebug()
    hybrid.load_weights(weight_path)
    hybrid.eval()

    # 创建STFT音频处理模块和梅尔滤波器组配置
    stft = model_def.STFT_Process(
        n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH
    ).eval()
    fbank = (
        (
            torchaudio.functional.melscale_fbanks(
                NFFT_STFT // 2 + 1,
                20,
                SAMPLE_RATE // 2,
                N_MELS,
                SAMPLE_RATE,
                None,
                "htk",
            )
        )
        .transpose(0, 1)
        .unsqueeze(0)
    )

    with torch.no_grad():
        # 封装模型以导出完整中间层输出，包含音频预处理和编码器各阶段
        enc_wrapper = model_def.EncoderExportWrapperDebug(hybrid, stft, fbank).eval()
        dummy_samples = SAMPLE_RATE * 1
        audio = torch.randn(1, 1, dummy_samples)
        ilens = torch.tensor([dummy_samples], dtype=torch.long)

        # results: (mel, lfr_x) + x, enc0, *debug_layers, enc_main, enc_tp + (enc_final, final_output)
        # Note: audio_encoder returns (enc_final, enc0, *debug_layers, enc_main, enc_tp)
        # Wrapper returns (mel, lfr_x, enc0, *debug_layers, enc_main, enc_tp, enc_final, final_output)
        output_names = (
            ["mel", "lfr_x", "enc0"]
            + [f"enc{i}" for i in range(1, 50)]
            + ["enc_main", "enc_tp", "enc_final", "final_output"]
        )

        # 导出ONNX模型，配置动态轴支持可变长度音频输入和批处理
        torch.onnx.export(
            enc_wrapper,
            (audio, ilens),
            onnx_fp32_path,
            input_names=["audio", "ilens"],
            output_names=output_names,
            dynamic_axes={
                "audio": {2: "samples"},
                "ilens": {0: "batch"},
                **{name: {1: f"{name}_len"} for name in output_names},
            },
            opset_version=OPSET,
            dynamo=True,
        )
    print(f"Saved FP32: {onnx_fp32_path}")

    print("\n--- Converting to FP16 (with FP32 LayerNorm) ---")
    # 加载FP32模型并转换为FP16精度，保留LayerNormalization层为FP32避免精度损失
    model = onnx.load(onnx_fp32_path)
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=False,
        op_block_list=[
            "LayerNormalization",
        ],
    )
    onnx.save(model_fp16, onnx_fp16_path)
    print(f"Saved FP16: {onnx_fp16_path}")


if __name__ == "__main__":
    main()
