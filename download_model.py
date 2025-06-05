import subprocess
import os

def download_and_convert_model():
    # 모델 다운로드 및 변환
    print("YOLO 모델 다운로드 및 변환을 시작합니다...")
    
    # 모델 다운로드 명령어
    download_cmd = "python3 -m depthai_downloader -m yolo11n.pt -o yolo11n.blob"
    
    try:
        subprocess.run(download_cmd, shell=True, check=True)
        print("모델 다운로드 및 변환이 완료되었습니다.")
        print("생성된 파일: yolo11n.blob")
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")
        print("모델 다운로드 및 변환에 실패했습니다.")

if __name__ == "__main__":
    download_and_convert_model() 