# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path="/Users/sonchansoo/Desktop/test/yolo11n1.tflite")
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Set up webcam
# cap = cv2.VideoCapture(0)  # Use 0 for built-in webcam or change to 1 for external webcam

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Preprocess the frame: resize and normalize
#     img = cv2.resize(frame, (224, 224))
#     img = img.astype(np.float32) / 255.0
#     input_data = np.expand_dims(img, axis=0)

#     # Perform inference
#     interpreter.set_tensor(input_details[0]['index'], input_data)
#     interpreter.invoke()

#     # Get the prediction result
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     prediction = output_data[0][0]

#     # Display the result on the frame
#     label = "Class 1" if prediction > 0.5 else "Class 0"
#     confidence = prediction if prediction > 0.5 else prediction
#     cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     # Show the frame with the prediction
#     cv2.imshow('Webcam', frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
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
