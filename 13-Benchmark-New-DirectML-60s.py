import onnxruntime as ort
import numpy as np
import time
import os
import json
from collections import defaultdict

def analyze_profile(profile_file):
    with open(profile_file, 'r') as f:
        data = json.load(f)
    
    op_stats = defaultdict(float)
    for entry in data:
        if 'cat' in entry and entry['cat'] in ['Op', 'Node'] and 'dur' in entry:
            op_name = entry.get('args', {}).get('op_name', entry['name'])
            op_stats[op_name] += entry['dur']
            
    sorted_stats = sorted(op_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"\n--- Operator Statistics (Top 10) ---")
    print(f"{'Operator':<25} | {'Total Time (ms)':<15}")
    print("-" * 45)
    for name, dur in sorted_stats[:10]:
        print(f"{name:<25} | {dur/1000:<15.2f}")

def benchmark_new_model_dml():
    model_path = r'./export_onnx_paddable/encoder_paddable.onnx'
    print(f"--- Profiling New Paddable Model (DirectML, 60s) ---")

    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True  # Enable Profiling
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, sess_opts, providers=providers)
    
    input_samples = 16000 * 60
    feed = {
        'audio': np.random.randn(1, 1, input_samples).astype(np.float32),
        'ilens': np.array([input_samples], dtype=np.int64)
    }

    print("[1] Warming up...")
    session.run(None, feed)

    print("[2] Running Profiling Run...")
    start = time.perf_counter()
    session.run(None, feed)
    print(f"    Single run took: {(time.perf_counter() - start)*1000:.2f} ms")

    profile_file = session.end_profiling()
    analyze_profile(profile_file)
    print(f"\nProfile saved to: {profile_file}")

if __name__ == "__main__":
    benchmark_new_model_dml()
