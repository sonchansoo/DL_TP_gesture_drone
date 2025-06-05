import cv2
from ultralytics import YOLO

# 1) TFLite 모델 경로를 지정해 YOLO 객체 생성
model = YOLO("/Users/sonchansoo/Desktop/test/best1.pt")

# 2) 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 3) YOLO(TFLite) 예측 → results 에는 검출된 객체들 정보가 들어있음
    results = model(frame)  # frame을 넘기면 내부에서 640×640 리사이징, 정규화, NMS까지 전부 처리

    # 4) 검출 결과 시각화
    annotated_frame = results[0].plot()  
    #   results[0].boxes, results[0].masks, results[0].keypoints 등을 plot()으로 함께 그려줌

    cv2.imshow("YOLO TFLite Detect", annotated_frame)

    # 5) 'q' 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()