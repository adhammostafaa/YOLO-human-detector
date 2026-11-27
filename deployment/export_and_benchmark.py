import os
import argparse
from ultralytics import YOLO


def get_size_mb(path):
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            total += os.path.getsize(os.path.join(root, fn))
    return round(total / (1024 * 1024), 2)


def run_benchmark(pt_path, yaml_file, imgsz=640):
    base_model = YOLO(pt_path)
    onnx_path = base_model.export(format='onnx', imgsz=imgsz)
    ov_int8_path = base_model.export(format='openvino', imgsz=imgsz, int8=True, data=yaml_file)

    variants = [
        ('PyTorch FP32 (.pt)', pt_path),
        ('ONNX FP32 (.onnx)', onnx_path),
        ('OpenVINO INT8', ov_int8_path)
    ]

    for name, path in variants:
        m = YOLO(path, task='detect')
        res = m.val(data=yaml_file, imgsz=imgsz, device='cpu', batch=1, verbose=False)
        ms = res.speed['inference']
        fps = 1000.0 / ms
        size_mb = get_size_mb(path)
        print(f"{name:20s} | size: {size_mb:6.2f} MB | speed: {ms:6.1f} ms ({fps:5.1f} FPS) | mAP50: {res.box.map50:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLO model to ONNX & OpenVINO INT8 and benchmark on CPU')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained .pt weights')
    parser.add_argument('--data', type=str, default='configs/drone_human.yaml', help='Path to dataset YAML for INT8 calibration & val')
    parser.add_argument('--imgsz', type=int, default=640, help='Export and benchmark input resolution')
    args = parser.parse_args()
    run_benchmark(args.weights, args.data, args.imgsz)
