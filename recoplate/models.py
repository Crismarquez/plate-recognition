import numpy as np
from typing import Dict, List, Tuple

from recoplate.detection import PlateDetection, YoloPlateDetection
from recoplate.recognition import OCRMotorModel
from recoplate.utils import crop_center, load_tf_model

ALLOW_YOLO_DETECTOR_MODELS = ["onnx", "pt"]

class PlateRecognition:
    def __init__(self, detector_name: str, ocr_method:str) -> None:
        
        self.detector_name = detector_name
        self.ocr_method = ocr_method

        if detector_name in ALLOW_YOLO_DETECTOR_MODELS:
            self.detection = YoloPlateDetection(detector_name)
        else:
            self.detection = PlateDetection(detector_name)
        self.recognition = OCRMotorModel(ocr_method)
        
    def _predict(self, frame: np.ndarray) -> Dict:

        predict_data = {}

        if self.detector_name in ALLOW_YOLO_DETECTOR_MODELS:
            cropped_plate, detections = self.detection.predict(frame)
        else:
            cropped_plate, detections = self.detection.predict(frame, self.object_detection_trh)

        all_plate_text = []
        scores = []
        for crop_plate in cropped_plate:
            plate_text, score = self.recognition.predict(crop_plate, self.ocr_threshold)
            all_plate_text.append(plate_text)
            scores.append(score)
        
        predict_data["detection"] = (cropped_plate, detections)
        predict_data["recognition"] = (all_plate_text, scores)

        return (cropped_plate, all_plate_text, predict_data)

    def predict(
        self, frame: np.ndarray, object_detection_trh: float=0.8,  ocr_threshold: float=0.5
        ) -> List:

        self.object_detection_trh = object_detection_trh
        self.ocr_threshold = ocr_threshold

        cropped_plate, all_plate_text, _ = self._predict(frame)

        return [cropped_plate, all_plate_text]


class PlateRecognitionLite:

    def __init__(self, ocr_method) -> None:
        self.ocr_method = ocr_method

        self.recognition = OCRMotorModel(ocr_method)

    def _predict(self, frame):
        cropped_plate = [crop_center(frame, self.object_detection_range)]

        predict_data = {}
        all_plate_text = []
        scores = []
        for crop_plate in cropped_plate:
            plate_text, score = self.recognition.predict(crop_plate, self.ocr_threshold)
            all_plate_text.append(plate_text)
            scores.append(score)

        predict_data["detection"] = (cropped_plate)
        predict_data["recognition"] = (all_plate_text, scores)

        return (cropped_plate, all_plate_text, predict_data)



    def predict(self, frame, object_detection_range:List=[200, 300] ,ocr_threshold: float=0.5):
        self.object_detection_range = object_detection_range
        self.ocr_threshold = ocr_threshold

        cropped_plate, all_plate_text, _ = self._predict(frame)
        
        return [cropped_plate, all_plate_text]

