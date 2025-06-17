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
        self.model = YOLO("/Users/sonchansoo/Desktop/test/best_float32.tflite")
        
        # 노트북 카메라 설정
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 제어 파라미터
        self.center_x = 640 // 2
        self.center_y = 480 // 2
        self.speed = 30

    def move_forward_up(self):
        """전진하면서 상승"""
        try:
            print("전진 및 상승 시작")
            self.drone.send_rc_control(0, self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_up(self):
        """후진하면서 상승"""
        try:
            print("후진 및 상승 시작")
            self.drone.send_rc_control(0, -self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_left_up(self):
        """좌측 이동하면서 상승"""
        try:
            print("좌측 이동 및 상승 시작")
            self.drone.send_rc_control(-self.speed, 0, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_right_up(self):
        """우측 이동하면서 상승"""
        try:
            print("우측 이동 및 상승 시작")
            self.drone.send_rc_control(self.speed, 0, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_forward_down(self):
        """전진하면서 하강"""
        try:
            print("전진 및 하강 시작")
            self.drone.send_rc_control(0, self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_down(self):
        """후진하면서 하강"""
        try:
            print("후진 및 하강 시작")
            self.drone.send_rc_control(0, -self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_left_down(self):
        """좌측 이동하면서 하강"""
        try:
            print("좌측 이동 및 하강 시작")
            self.drone.send_rc_control(-self.speed, 0, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_right_down(self):
        """우측 이동하면서 하강"""
        try:
            print("우측 이동 및 하강 시작")
            self.drone.send_rc_control(self.speed, 0, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_forward_left_up(self):
        """전진하면서 좌측 이동 및 상승"""
        try:
            print("전진, 좌측 이동 및 상승 시작")
            self.drone.send_rc_control(-self.speed, self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_left_up(self):
        """후진하면서 좌측 이동 및 상승"""
        try:
            print("후진, 좌측 이동 및 상승 시작")
            self.drone.send_rc_control(-self.speed, -self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_forward_right_up(self):
        """전진하면서 우측 이동 및 상승"""
        try:
            print("전진, 우측 이동 및 상승 시작")
            self.drone.send_rc_control(self.speed, self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_right_up(self):
        """후진하면서 우측 이동 및 상승"""
        try:
            print("후진, 우측 이동 및 상승 시작")
            self.drone.send_rc_control(self.speed, -self.speed, self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_forward_left_down(self):
        """전진하면서 좌측 이동 및 하강"""
        try:
            print("전진, 좌측 이동 및 하강 시작")
            self.drone.send_rc_control(-self.speed, self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_forward_right_down(self):
        """전진하면서 우측 이동 및 하강"""
        try:
            print("전진, 우측 이동 및 하강 시작")
            self.drone.send_rc_control(self.speed, self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_left_down(self):
        """후진하면서 좌측 이동 및 하강"""
        try:
            print("후진, 좌측 이동 및 하강 시작")
            self.drone.send_rc_control(-self.speed, -self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def move_back_right_down(self):
        """후진하면서 우측 이동 및 하강"""
        try:
            print("후진, 우측 이동 및 하강 시작")
            self.drone.send_rc_control(self.speed, -self.speed, -self.speed, 0)
            time.sleep(2)
            self.drone.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"동작 실행 중 오류 발생: {e}")
            self.drone.send_rc_control(0, 0, 0, 0)

    def get_frame(self):
        """노트북 카메라에서 프레임 가져오기"""
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    # 클래스별 동작 매핑
    class_actions = {
        0: lambda self: self.drone.takeoff(),      # A: 이륙
        1: lambda self: self.drone.land(),         # B: 착륙
        2: lambda self: self.drone.move_forward(self.speed),  # C: 전진
        3: lambda self: self.drone.move_back(self.speed),     # D: 후진
        4: lambda self: self.drone.move_left(self.speed),     # E: 좌이동
        5: lambda self: self.drone.move_right(self.speed),    # F: 우이동
        6: lambda self: self.drone.move_up(self.speed),       # G: 상승
        7: lambda self: self.drone.move_down(self.speed),     # H: 하강
        8: lambda self: self.drone.rotate_clockwise(90),      # I: 시계방향 회전
        9: lambda self: self.drone.rotate_counter_clockwise(90),  # J: 반시계방향 회전
        10: lambda self: self.move_forward_up(),           # K: 전방+상승
        11: lambda self: self.move_back_up(),              # L: 후방+상승
        12: lambda self: self.move_left_up(),              # M: 좌측+상승
        13: lambda self: self.move_right_up(),             # N: 우측+상승
        14: lambda self: self.move_forward_down(),         # O: 하강+전진
        15: lambda self: self.move_back_down(),            # P: 하강+후진
        16: lambda self: self.move_left_down(),            # Q: 하강+좌이동
        17: lambda self: self.move_right_down(),           # R: 하강+우이동
        18: lambda self: self.move_forward_left_up(),      # S: 전진+좌이동+상승
        19: lambda self: self.move_back_left_up(),         # T: 후진+좌이동+상승
        20: lambda self: self.move_forward_right_up(),     # U: 전진+우이동+상승
        21: lambda self: self.move_back_right_up(),        # V: 후진+우이동+상승
        22: lambda self: self.move_forward_left_down(),    # W: 하강+전진+좌이동
        23: lambda self: self.move_forward_right_down(),   # X: 하강+전진+우이동
        24: lambda self: self.move_back_left_down(),       # Y: 하강+후진+좌이동
        25: lambda self: self.move_back_right_down()       # Z: 하강+후진+우이동
    }
    
    # 클래스 이름 매핑
    class_names = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
        24: "Y",
        25: "Z"
    }

    def execute_action(self, class_id):
        """감지된 클래스에 따른 동작 실행"""
        if class_id in self.class_actions:
            try:
                self.class_actions[class_id](self)
                print(f"실행된 동작: {self.class_names[class_id]} - {self.get_action_description(class_id)}")
            except Exception as e:
                print(f"동작 실행 중 오류 발생: {e}")

    def get_action_description(self, class_id):
        """클래스 ID에 따른 동작 설명 반환"""
        descriptions = {
            0: "이륙",
            1: "착륙",
            2: "전진",
            3: "후진",
            4: "좌이동",
            5: "우이동",
            6: "상승",
            7: "하강",
            8: "시계방향 회전",
            9: "반시계방향 회전",
            10: "전방+상승",
            11: "후방+상승",
            12: "좌측+상승",
            13: "우측+상승",
            14: "하강+전진",
            15: "하강+후진",
            16: "하강+좌이동",
            17: "하강+우이동",
            18: "전진+좌이동+상승",
            19: "후진+좌이동+상승",
            20: "전진+우이동+상승",
            21: "후진+우이동+상승",
            22: "하강+전진+좌이동",
            23: "하강+전진+우이동",
            24: "하강+후진+좌이동",
            25: "하강+후진+우이동"
        }
        return descriptions.get(class_id, "알 수 없는 동작")

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
            print("K: 전방+상승, L: 후방+상승")
            print("M: 좌측+상승, N: 우측+상승")
            print("O: 하강+전진, P: 하강+후진")
            print("Q: 하강+좌이동, R: 하강+우이동")
            print("S: 전진+좌이동+상승, T: 후진+좌이동+상승")
            print("U: 전진+우이동+상승, V: 후진+우이동+상승")
            print("W: 하강+전진+좌이동, X: 하강+전진+우이동")
            print("Y: 하강+후진+좌이동, Z: 하강+후진+우이동")
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
            self.drone.end()
            self.cap.release()
            cv2.destroyAllWindows()
            print("프로그램 종료")

if __name__ == "__main__":
    controller = TelloYOLOController()
    controller.run() 