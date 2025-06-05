import cv2
import depthai as dai

def main_oakd():
    # DepthAI 파이프라인 생성
    pipeline = dai.Pipeline()

    # 컬러 카메라 노드 생성
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)  # 미리보기 해상도 설정 (원하는 크기로 조절 가능)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # OpenCV와 호환되도록 BGR 순서로 설정
    cam_rgb.setFps(30)

    # XLinkOut 노드 생성 (호스트로 데이터 전송)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # 파이프라인을 OAK-D 장치에 연결하고 시작
    try:
        with dai.Device(pipeline) as device:
            print("OAK-D 카메라를 성공적으로 열었습니다.")
            print("장치 이름:", device.getMxId()) # 장치 ID 출력 (디버깅에 유용)
            if device.getBootloaderVersion() is not None:
                print("부트로더 버전:", device.getBootloaderVersion())
            if device.getConnectedCameraFeatures():
                print("연결된 카메라:", device.getConnectedCameraFeatures())


            # RGB 스트림을 위한 출력 큐 가져오기
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            print("'q' 키를 누르면 종료됩니다.")

            while True:
                in_rgb = q_rgb.tryGet()  # 논블로킹 방식으로 데이터 가져오기

                if in_rgb is not None:
                    # ImgFrame을 OpenCV Mat으로 변환
                    frame = in_rgb.getCvFrame()

                    # 프레임을 'OAK-D Camera Test' 창에 보여주기
                    cv2.imshow("OAK-D Camera Test", frame)

                # 'q' 키가 눌리면 루프 종료
                if cv2.waitKey(1) == ord('q'):
                    print("종료 키 입력됨. 프로그램을 종료합니다.")
                    break

    except RuntimeError as e:
        print(f"OAK-D 장치 오류: {e}")
        print("OAK-D 카메라가 연결되어 있는지, 다른 프로그램에서 사용 중이지 않은지 확인해주세요.")
        print("Linux 사용자의 경우 udev 규칙이 올바르게 설정되었는지 확인하세요.")
        print("   (참고: https://docs.luxonis.com/projects/api/en/latest/install/#udev-rules-for-usb-permissions)")


    # 모든 OpenCV 창 닫기
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_oakd()