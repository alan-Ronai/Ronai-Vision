from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        results = self.model(frame, device=self.device)[0]
        return results
