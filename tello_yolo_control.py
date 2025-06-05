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
            0: self.drone.takeoff,      # A: 이륙
            1: self.drone.land,         # B: 착륙
            2: lambda: self.drone.move_forward(self.speed),  # C: 전진
            3: lambda: self.drone.move_back(self.speed),     # D: 후진
            4: lambda: self.drone.move_left(self.speed),     # E: 좌이동
            5: lambda: self.drone.move_right(self.speed),    # F: 우이동
            6: lambda: self.drone.move_up(self.speed),       # G: 상승
            7: lambda: self.drone.move_down(self.speed),     # H: 하강
            8: lambda: self.drone.rotate_clockwise(90),      # I: 시계방향 회전
            9: lambda: self.drone.rotate_counter_clockwise(90),  # J: 반시계방향 회전
            10: lambda: self.drone.flip_forward(),           # K: 전방 회전
            11: lambda: self.drone.flip_back(),              # L: 후방 회전
            12: lambda: self.drone.flip_left(),              # M: 좌측 회전
            13: lambda: self.drone.flip_right(),             # N: 우측 회전
            14: lambda: self.drone.move_forward(self.speed * 2),  # O: 빠른 전진
            15: lambda: self.drone.move_back(self.speed * 2),     # P: 빠른 후진
            16: lambda: self.drone.move_left(self.speed * 2),     # Q: 빠른 좌이동
            17: lambda: self.drone.move_right(self.speed * 2),    # R: 빠른 우이동
            18: lambda: self.drone.move_up(self.speed * 2),       # S: 빠른 상승
            19: lambda: self.drone.move_down(self.speed * 2),     # T: 빠른 하강
            20: lambda: self.drone.rotate_clockwise(180),         # U: 180도 회전
            21: lambda: self.drone.rotate_counter_clockwise(180), # V: 180도 반회전
            22: lambda: self.drone.move_forward(self.speed * 3),  # W: 매우 빠른 전진
            23: lambda: self.drone.move_back(self.speed * 3),     # X: 매우 빠른 후진
            24: lambda: self.drone.move_left(self.speed * 3),     # Y: 매우 빠른 좌이동
            25: lambda: self.drone.move_right(self.speed * 3),    # Z: 매우 빠른 우이동
        }
        
        # 클래스 이름 매핑
        self.class_names = {
            0: "A (이륙)",
            1: "B (착륙)",
            2: "C (전진)",
            3: "D (후진)",
            4: "E (좌이동)",
            5: "F (우이동)",
            6: "G (상승)",
            7: "H (하강)",
            8: "I (시계방향 회전)",
            9: "J (반시계방향 회전)",
            10: "K (전방 회전)",
            11: "L (후방 회전)",
            12: "M (좌측 회전)",
            13: "N (우측 회전)",
            14: "O (빠른 전진)",
            15: "P (빠른 후진)",
            16: "Q (빠른 좌이동)",
            17: "R (빠른 우이동)",
            18: "S (빠른 상승)",
            19: "T (빠른 하강)",
            20: "U (180도 회전)",
            21: "V (180도 반회전)",
            22: "W (매우 빠른 전진)",
            23: "X (매우 빠른 후진)",
            24: "Y (매우 빠른 좌이동)",
            25: "Z (매우 빠른 우이동)"
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
            print("수화 알파벳에 따라 드론이 자동으로 동작합니다.")
            print("A: 이륙, B: 착륙")
            print("C: 전진, D: 후진")
            print("E: 좌이동, F: 우이동")
            print("G: 상승, H: 하강")
            print("I: 시계방향 회전, J: 반시계방향 회전")
            print("K: 전방 회전, L: 후방 회전")
            print("M: 좌측 회전, N: 우측 회전")
            print("O: 빠른 전진, P: 빠른 후진")
            print("Q: 빠른 좌이동, R: 빠른 우이동")
            print("S: 빠른 상승, T: 빠른 하강")
            print("U: 180도 회전, V: 180도 반회전")
            print("W: 매우 빠른 전진, X: 매우 빠른 후진")
            print("Y: 매우 빠른 좌이동, Z: 매우 빠른 우이동")
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