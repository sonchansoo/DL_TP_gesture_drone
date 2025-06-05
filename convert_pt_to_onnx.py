from ultralytics import YOLO
import torch
import os

def convert_pt_to_onnx(pt_path, onnx_path):
    print(f"변환 시작: {pt_path} -> {onnx_path}")
    
    # YOLO 모델 로드
    model = YOLO(pt_path)
    
    # ONNX로 변환 (OAK-D 호환성을 위한 설정 추가)
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    print(f"ONNX 변환 완료: {onnx_path}")

if __name__ == "__main__":
    # 변환할 .pt 파일 경로
    pt_file = "/Users/sonchansoo/Desktop/test/yolo11n.pt"
    if not pt_file.endswith('.pt'):
        print("올바른 .pt 파일을 입력해주세요.")
        exit(1)
    
    # ONNX 파일 경로 생성
    onnx_file = pt_file.replace('.pt', '.onnx')
    
    try:
        convert_pt_to_onnx(pt_file, onnx_file)
        print("변환이 성공적으로 완료되었습니다!")
        print(f"ONNX 파일이 생성되었습니다: {onnx_file}")
        print("\n다음 단계:")
        print("1. OAK-D의 model-compiler 도구를 사용하여 ONNX를 blob으로 변환하세요.")
        print("2. 변환 명령어: model-compiler -m yolo11n.onnx -o yolo11n.blob")
    except Exception as e:
        print(f"변환 중 오류 발생: {e}") 