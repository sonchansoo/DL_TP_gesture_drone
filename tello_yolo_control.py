import cv2
from ultralytics import YOLO
from djitellopy import Tello
import time
import numpy as np

class TelloYOLOController:
    def __init__(self):
        # Tello 드론 초기화
        self.drone = Tello()
        self.drone.connect()
        print(f"배터리 잔량: {self.drone.get_battery()}%")
        
        # YOLO 모델 로드
        self.model = YOLO("/Users/sonchansoo/Desktop/test/best1.pt")
        
        # 노트북 카메라 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 제어 파라미터
        self.center_x = 640 // 2
        self.center_y = 480 // 2
        self.speed = 30
        
        # 클래스별 동작 매핑
        self.class_actions = {
            0: self.drone.takeoff,  # 클래스 0: 이륙
            1: self.drone.land,     # 클래스 1: 착륙
            2: lambda: self.drone.move_forward(self.speed),  # 클래스 2: 전진
            3: lambda: self.drone.move_back(self.speed),     # 클래스 3: 후진
            4: lambda: self.drone.move_left(self.speed),     # 클래스 4: 좌이동
            5: lambda: self.drone.move_right(self.speed),    # 클래스 5: 우이동
            6: lambda: self.drone.move_up(self.speed),       # 클래스 6: 상승
            7: lambda: self.drone.move_down(self.speed),     # 클래스 7: 하강
        }
        
        # 클래스 이름 매핑 (실제 학습된 클래스 이름으로 수정 필요)
        self.class_names = {
            0: "takeoff",
            1: "land",
            2: "forward",
            3: "back",
            4: "left",
            5: "right",
            6: "up",
            7: "down"
        }

    def get_frame(self):
        """노트북 카메라에서 프레임 가져오기"""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def execute_action(self, class_id):
        """감지된 클래스에 따른 동작 실행"""
        if class_id in self.class_actions:
            try:
                self.class_actions[class_id]()
                print(f"실행된 동작: {self.class_names[class_id]}")
            except Exception as e:
                print(f"동작 실행 중 오류 발생: {e}")

    def process_detection(self, frame):
        """YOLO로 객체 감지 및 결과 처리"""
        results = self.model(frame)
        annotated_frame = results[0].plot()
        
        # 감지된 객체가 있는지 확인
        if len(results[0].boxes) > 0:
            # 가장 큰 객체 찾기 (가장 큰 바운딩 박스)
            boxes = results[0].boxes
            max_area = 0
            target_box = None
            target_class = None
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    target_box = box
                    target_class = int(box.cls[0].cpu().numpy())
            
            if target_box is not None:
                # 타겟 객체의 중심점 계산
                x1, y1, x2, y2 = target_box.xyxy[0].cpu().numpy()
                target_x = int((x1 + x2) / 2)
                target_y = int((y1 + y2) / 2)
                
                # 중심점 표시
                cv2.circle(annotated_frame, (target_x, target_y), 5, (0, 255, 0), -1)
                
                # 감지된 클래스에 따른 동작 실행
                self.execute_action(target_class)
                
                # 상태 표시
                class_name = self.class_names.get(target_class, "unknown")
                cv2.putText(annotated_frame, f"Class: {class_name}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame

    def run(self):
        """메인 실행 루프"""
        try:
            print("프로그램 시작")
            print("감지된 객체에 따라 드론이 자동으로 동작합니다.")
            print("q: 종료")
            
            while True:
                # 프레임 가져오기
                frame = self.get_frame()
                if frame is None:
                    break
                
                # 객체 감지 및 드론 제어
                annotated_frame = self.process_detection(frame)
                
                # 화면 표시
                cv2.imshow("Tello YOLO Control", annotated_frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 안정성을 위한 짧은 대기
                time.sleep(0.1)
                
        except Exception as e:
            print(f"오류 발생: {e}")
        finally:
            # 정리
            self.drone.land()
            self.cap.release()
            cv2.destroyAllWindows()
            print("프로그램 종료")

if __name__ == "__main__":
    controller = TelloYOLOController()
    controller.run() 