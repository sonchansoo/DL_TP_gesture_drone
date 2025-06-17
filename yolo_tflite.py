import cv2
import numpy as np
import tensorflow as tf

# 클래스 이름 A~Z
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
model_path = "/Users/sonchansoo/Desktop/test/best_float32.tflite"  # float32 모델 사용

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 출력 형식 디버깅
print("입력 shape:", input_details[0]['shape'])
print("출력 shape:", output_details[0]['shape'])

# YOLOv8 모델의 입력 크기 (640x640)
INPUT_SIZE = 640
H, W = INPUT_SIZE, INPUT_SIZE

# 신뢰도 임계값 설정 (80%)
CONF_THRESHOLD = 0.80

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    # 전처리
    img = cv2.resize(frame, (W, H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = np.expand_dims(img, 0).astype(np.float32) / 255.0

    # 추론
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])[0]  # shape: (8400, 85)

    boxes, scores, ids = [], [], []
    xf, yf = fw / W, fh / H

    for p in out:
        # YOLOv8 출력 형식: [x, y, w, h, objectness, class_scores...]
        cx, cy, w, h = p[:4]
        objectness = p[4]
        class_scores = p[5:5+26]  # 26개 클래스 점수만 사용
        
        # 클래스 점수가 가장 높은 것 선택
        cid = int(np.argmax(class_scores))
        class_score = float(class_scores[cid])
        
        # 최종 신뢰도 = objectness * class_score
        conf = float(objectness * class_score)
        
        # 신뢰도가 임계값(80%) 이상인 경우에만 처리
        if conf >= CONF_THRESHOLD:
            # 원본 화면 크기로 복원
            x = int((cx - w/2) * xf)
            y = int((cy - h/2) * yf)
            w_pix = int(w * xf)
            h_pix = int(h * yf)

            boxes.append([x, y, w_pix, h_pix])
            scores.append(conf)
            ids.append(cid)

    # NMS: macOS OpenCV에서 tuple 반환 대응
    if len(boxes) == 0:
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ─── NMS 실행 ────────────────────────────────────────────
    raw_nms = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESHOLD, nms_threshold=0.45)

    # ─── tuple([indices]) 형태라면 첫 번째 요소로 꺼내고 ──────────
    if isinstance(raw_nms, tuple):
        raw_nms = raw_nms[0]

    # ─── numpy array나 리스트로 통일 후 flatten ─────────────
    idxs = np.array(raw_nms).flatten().astype(int)

    # ─── 바운딩 박스 그리기 ─────────────────────────────────
    for i in idxs:
        x, y, w_pix, h_pix = boxes[i]
        cid = ids[i]
        label = class_names[cid]
        conf = scores[i]
        
        # 바운딩 박스 그리기
        cv2.rectangle(frame, (x, y), (x + w_pix, y + h_pix), (0, 255, 0), 2)
        
        # 클래스 이름과 신뢰도 텍스트 만들기
        label_text = f"{label}: {conf:.2f}"
        
        # 텍스트 배경 크기 계산
        (label_w, label_h), _ = cv2.getTextSize(label_text, 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # 텍스트 배경 그리기
        cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
        
        # 텍스트 그리기
        cv2.putText(frame, label_text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # ─── 화면에 보여주기 ─────────────────────────────────────
    cv2.imshow("YOLO Detection (Conf > 80%)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
