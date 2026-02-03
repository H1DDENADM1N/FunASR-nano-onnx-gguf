import onnxruntime as ort
import numpy as np
import time
import os

def benchmark_new_fp16_30s():
    model_path = r'./export_onnx_paddable/encoder_paddable.fp16.onnx'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    print(f"--- Benchmark: New Paddable Model FP16 (DirectML, 30s) ---")
    print(f"Model: {model_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = [
        ('DmlExecutionProvider'),
        'CPUExecutionProvider'
    ]
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    
    out_names = [x.name for x in session.get_outputs()]
    
    # Prepare 30s OrtValues
    SAMPLE_RATE = 16000
    SEC = 30
    input_samples = SAMPLE_RATE * SEC
    
    # FP16 model still takes Float32 input in many cases unless explicitly cast, 
    # but ORT usually handles the conversion if keep_io_types was False.
    # We check input type to be sure.
    input_type = session.get_inputs()[0].type
    print(f"Input Type: {input_type}")
    
    if "float16" in input_type:
        audio_np = np.random.randn(1, 1, input_samples).astype(np.float16)
    else:
        audio_np = np.random.randn(1, 1, input_samples).astype(np.float32)
        
    ilens_np = np.array([input_samples], dtype=np.int64)
    
    audio_ort = ort.OrtValue.ortvalue_from_numpy(audio_np, 'cpu', 0)
    ilens_ort = ort.OrtValue.ortvalue_from_numpy(ilens_np, 'cpu', 0)
    
    feed = {'audio': audio_ort, 'ilens': ilens_ort}

    print(f"[1] Warming up ({SEC}s input)...")
    session.run_with_ort_values(out_names, feed)

    print(f"[2] Running 3 consecutive benchmarks (30s each)...")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        session.run_with_ort_values(out_names, feed)
        times.append((time.perf_counter() - t0) * 1000)
        print(f"    Run {i+1}: {times[-1]:.2f} ms")

    avg_lat = sum(times)/3
    print(f"\nAverage Latency: {avg_lat:.2f} ms")
    print(f"Real-time factor: {avg_lat / (SEC * 1000):.4f}x (Lower is better)")

if __name__ == "__main__":
    benchmark_new_fp16_30s()
