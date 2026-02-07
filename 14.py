import numpy as np
import onnxruntime as ort

# Configuration
ONNX_FP32 = "./model_debug/debug_fp32.onnx"
ONNX_FP16 = "./model_debug/debug_fp16.onnx"
DEVICE_ID = 0


def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度，用于比较模型输出的相似性。

    参数:
        a (numpy.ndarray): 第一个输入向量（任意形状张量）
        b (numpy.ndarray): 第二个输入向量（任意形状张量）

    返回:
        float: 余弦相似度值（范围[-1,1]），当向量模乘积接近零时返回1.0避免除零错误
    """
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    if norm_product < 1e-9:
        return 1.0
    return np.dot(a, b) / (norm_product + 1e-9)


def compare_outputs(out32_dict, out16_dict):
    """
    比较FP32和FP16模型各层输出的差异，生成详细的对比报告。

    参数:
        out32_dict (dict): FP32模型输出字典，键为层名称，值为输出张量
        out16_dict (dict): FP16模型输出字典，键为层名称，值为输出张量

    返回:
        None: 直接打印对比结果表格，遇到关键错误时提前终止
    """
    print("\n" + "=" * 110)
    print(
        f"{'Layer Name':<20} | {'Cos Sim':<10} | {'Max Abs Err':<12} | {'Max Val(32)':<12} | {'Max Val(16)':<12}"
    )
    print("-" * 110)

    # 预定义需要比较的层列表（与13.py中的层结构严格对应）
    LAYERS = (
        ["mel", "lfr_x", "enc0"]
        + [f"enc{i}" for i in range(1, 50)]
        + ["enc_main", "enc_tp", "enc_final", "final_output"]
    )

    # 遍历预定义层列表，逐层比较输出差异
    for name in LAYERS:
        if name not in out32_dict or name not in out16_dict:
            continue

        v32 = out32_dict[name]
        v16 = out16_dict[name]

        sim = cosine_similarity(v32, v16)
        max_err = np.max(np.abs(v32.astype(np.float32) - v16.astype(np.float32)))
        max_v32 = np.max(np.abs(v32.astype(np.float32)))
        max_v16 = np.max(np.abs(v16.astype(np.float32)))

        status = ""
        if sim < 0.99 or max_err > 50.0:
            status = " <<< ALERT!"
        if np.isnan(v16).any() or np.isinf(v16).any():
            status = " [CRITICAL NaN/Inf!]"

        print(
            f"{name:<20} | {sim:<10.6f} | {max_err:<12.6f} | {max_v32:<12.6f} | {max_v16:<12.6f} {status}"
        )

        if "CRITICAL" in status:
            print(f"Stopping at first critical failure layer: {name}")
            break

    print("=" * 110)


def main():
    """
    主执行函数：加载FP32/FP16模型，运行推理并比较输出差异。

    执行流程：
    1. 配置ONNX Runtime执行提供者（DML + CPU）
    2. 加载FP32和FP16模型
    3. 生成随机测试输入（2秒音频，16kHz采样率）
    4. 分别执行FP32和FP16模型推理
    5. 调用compare_outputs进行输出对比

    返回:
        None: 执行完成后直接退出
    """
    print(f"Loading Models with DML (device_id={DEVICE_ID})...")

    # 配置ONNX Runtime执行提供者（优先使用DML，回退到CPU）
    providers = [
        ("DmlExecutionProvider", {"device_id": DEVICE_ID}),
        "CPUExecutionProvider",
    ]

    try:
        sess32 = ort.InferenceSession(ONNX_FP32, providers=providers)
        sess16 = ort.InferenceSession(ONNX_FP16, providers=providers)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"Active Provider (FP32): {sess32.get_providers()[0]}")
    print(f"Active Provider (FP16): {sess16.get_providers()[0]}")

    # 生成随机测试输入（2秒音频，16kHz采样率）
    samples = 16000 * 2
    audio_32 = np.random.randn(1, 1, samples).astype(np.float32)
    ilens = np.array([samples], dtype=np.int64)
    audio_16 = audio_32.astype(np.float16)

    print("\nRunning Inference...")

    # 执行FP32模型推理
    v32_outs = sess32.run(None, {"audio": audio_32, "ilens": ilens})
    out_names = [x.name for x in sess32.get_outputs()]
    out32_dict = dict(zip(out_names, v32_outs))

    # 执行FP16模型推理（输入转换为float16）
    v16_outs = sess16.run(None, {"audio": audio_16, "ilens": ilens})
    out16_dict = dict(zip(out_names, v16_outs))

    # 比较并输出模型差异报告
    compare_outputs(out32_dict, out16_dict)


if __name__ == "__main__":
    main()
