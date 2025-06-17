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
import numpy as np
import tensorflow as tf
import os

# 혼동 행렬 이미지에서 클래스 이름 목록을 추출하여 정의합니다.
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# TFLite 모델 경로 (상대 경로)
model_path = "/Users/sonchansoo/Desktop/test/best_float32.tflite"

# 모델 로딩
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("모델을 성공적으로 로드했습니다.")

except Exception as e:
    print(f"모델 로딩에 실패했습니다: {e}")
    exit()

# 입력 및 출력 세부 정보를 가져옵니다.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 이미지 크기를 모델에 맞게 가져옵니다.
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# 웹캠을 설정합니다.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져오는 데 실패했습니다.")
        break

    # 프레임을 전처리합니다: 크기 조정 및 정규화
    img = cv2.resize(frame, (input_width, input_height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_data = np.expand_dims(img_rgb, axis=0)

    # 추론을 수행합니다.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 예측 결과를 가져옵니다.
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # 가장 높은 확률을 가진 클래스의 인덱스를 찾습니다.
    predicted_index = np.argmax(output_data)
    
    # 해당 클래스의 이름과 신뢰도(확률)를 가져옵니다.
    label = class_names[predicted_index]
    confidence = output_data[predicted_index]

    # 프레임에 결과를 표시합니다.
    display_text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 예측 결과가 있는 프레임을 보여줍니다.
    cv2.imshow('Webcam Classification', frame)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()

