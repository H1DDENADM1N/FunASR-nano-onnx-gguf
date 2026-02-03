import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType

# Configuration for the new Paddable Model
INPUT_DIR = "./export_onnx_paddable"
INPUT_MODEL = f"{INPUT_DIR}/encoder_paddable.onnx"

def convert_to_fp16(input_path):
    output_path = input_path.replace(".onnx", ".fp16.onnx")
    print(f"\n[FP16] Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        model = onnx.load(input_path)
        # Use ORT Transformers conversion for better DML compatibility
        # Parameters inspired by 02-Quantize-ONNX.py
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=False,
            min_positive_val=1e-7,
            max_finite_val=65504,
            op_block_list=[]
        )
        onnx.save(model_fp16, output_path)
        print(f"   [Success] Saved FP16 model.")
    except Exception as e:
        print(f"   [Failed] FP16 conversion error: {e}")

def convert_to_int8(input_path):
    output_path = input_path.replace(".onnx", ".int8.onnx")
    print(f"\n[INT8] Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            op_types_to_quantize=["MatMul"], # Primary target for weight compression
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"   [Success] Saved INT8 model.")
    except Exception as e:
        print(f"   [Failed] INT8 quantization error: {e}")

def main():
    print("--- Starting Quantization for Paddable Model ---")
    
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: Input model {INPUT_MODEL} not found.")
        return

    # 1. Convert to FP16 (Best for DirectML/GPU)
    convert_to_fp16(INPUT_MODEL)
    
    # 2. Convert to INT8 (Best for CPU/Mobile)
    convert_to_int8(INPUT_MODEL)

    print("\n--- All Quantizations Complete ---")

if __name__ == "__main__":
    main()
