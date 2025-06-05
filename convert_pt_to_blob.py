import torch
import torchvision
from openvino.runtime import Core
import os

def convert_pt_to_blob(pt_path, blob_path):
    print(f"변환 시작: {pt_path} -> {blob_path}")
    
    # 1. PyTorch 모델 로드
    model = torch.load(pt_path, weights_only=False)
    model.eval()
    
    # 2. 더미 입력 생성
    dummy_input = torch.randn(1, 3, 640, 640)
    
    # 3. ONNX로 변환
    onnx_path = pt_path.replace('.pt', '.onnx')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print(f"ONNX 변환 완료: {onnx_path}")
    
    # 4. ONNX를 OpenVINO IR로 변환
    core = Core()
    model_ov = core.read_model(onnx_path)
    
    # 5. OpenVINO IR을 blob으로 변환
    compiled_model = core.compile_model(model_ov, "MYRIAD")
    compiled_model.export_model(blob_path)
    print(f"BLOB 변환 완료: {blob_path}")
    
    # 임시 ONNX 파일 삭제
    os.remove(onnx_path)
    print("임시 파일 정리 완료")

if __name__ == "__main__":
    # 변환할 .pt 파일 경로를 입력받음
    pt_file = "/Users/sonchansoo/Desktop/test/yolo11n.pt"
    if not pt_file.endswith('.pt'):
        print("올바른 .pt 파일을 입력해주세요.")
        exit(1)
    
    # blob 파일 경로 생성
    blob_file = pt_file.replace('.pt', '.blob')
    
    try:
        convert_pt_to_blob(pt_file, blob_file)
        print("변환이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"변환 중 오류 발생: {e}") 