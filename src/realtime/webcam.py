import cv2


class Webcam:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Webcam has not been started.")

        success, frame = self.cap.read()

        if not success:
            raise RuntimeError("Could not read frame from webcam.")

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    webcam = Webcam()

    try:
        webcam.start()

        while True:
            frame = webcam.read()

            cv2.imshow("Waste Classification - Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        webcam.release()
        cv2.destroyAllWindows()