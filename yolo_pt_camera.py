import cv2
from ultralytics import YOLO
import numpy as np # numpy를 import 합니다.

# 1) 모델 경로를 지정해 YOLO 객체 생성
# TFLite 모델을 사용하려면 'best.tflite'와 같이 경로를 지정하면 됩니다.
# .pt 모델도 동일한 방식으로 로드할 수 있습니다.
model = YOLO("/Users/sonchansoo/Desktop/test/best_float32.tflite")

# 2) 웹캠 열기
cap = cv2.VideoCapture(0)

# 정확도(신뢰도) 임계값 설정 (80%)
CONF_THRESHOLD = 0.80

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 3) YOLO 예측
    results = model(frame)

    # 4) 검출 결과 수동으로 시각화 (정확도 필터링)
    # results[0].plot() 대신 직접 그리는 로직으로 변경합니다.
    
    # 첫 번째 이미지의 결과 가져오기
    result = results[0]
    
    # 클래스 이름 딕셔너리 가져오기
    class_names = result.names

    # 검출된 각 객체(box)에 대해 반복
    for box in result.boxes:
        # 객체의 신뢰도(confidence) 확인
        confidence = box.conf[0]

        # 신뢰도가 설정한 임계값(80%) 이상인 경우에만 그리기
        if confidence >= CONF_THRESHOLD:
            # 바운딩 박스 좌표 가져오기 (xyxy 형식)
            # .cpu()는 GPU 사용 시 결과를 CPU로 옮기기 위함, .numpy()는 텐서를 numpy 배열로 변환
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            # 클래스 ID 및 이름 가져오기
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # 프레임에 바운딩 박스 그리기
            # cv2.rectangle(이미지, 시작점, 끝점, 색상, 두께)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 클래스 이름과 신뢰도 텍스트 만들기
            label = f"{class_name}: {confidence:.2f}"
            
            # 프레임에 텍스트 표시
            # cv2.putText(이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5) 결과가 그려진 프레임 보여주기
    cv2.imshow("YOLO Detect (Conf > 80%)", frame)

    # 6) 'q' 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()