import onnxruntime as ort
import numpy as np
import time
import os

def benchmark_old_final():
    model_path = r'./model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return
        
    print(f"--- Final Benchmark: Old Model (DirectML, 60s) ---")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    
    input_name = session.get_inputs()[0].name
    out_names = [x.name for x in session.get_outputs()]
    
    # Prepare 60s OrtValue
    audio_np = np.random.randn(1, 1, 16000 * 60).astype(np.float32)
    audio_ort = ort.OrtValue.ortvalue_from_numpy(audio_np, 'cpu', 0)
    feed = {input_name: audio_ort}

    print("[1] Warming up (60s input)...")
    session.run_with_ort_values(out_names, feed)

    print("[2] Running 3 consecutive benchmarks...")
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        session.run_with_ort_values(out_names, feed)
        times.append((time.perf_counter() - t0) * 1000)
        print(f"    Run {i+1}: {times[-1]:.2f} ms")

    print(f"\nAverage Latency: {sum(times)/3:.2f} ms")

if __name__ == "__main__":
    benchmark_old_final()
