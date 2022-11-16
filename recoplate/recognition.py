from typing import Tuple

import numpy as np
from paddleocr import PaddleOCR

from recoplate.utils import ocr_to_motorplate

ALLOW_OCR_METHODS = ["paddle"]

class OCRModel:

    def __init__(self, method_name):
        self.method_name = method_name

        if method_name not in ALLOW_OCR_METHODS:
            raise ValueError(
                f"model {method_name} is not implemented, try someone: {ALLOW_OCR_METHODS}"
                )

        self.load_model()

    def load_model(self):
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en', show_log = False)

    def preprocess(self, plate_text, score):
        if score < self.ocr_threshold:
            return ""
        
        return plate_text

    def _predict(self, frame: np.ndarray) -> Tuple:

        #TODO: val how many text recognition
        predictions = self.ocr_model.ocr(frame, cls=False, det=True)
        if len(predictions)==0:
            plate_text, score = "", 0
            return plate_text, score 
            
        plate_text, score = predictions[0][1]
        plate_text = self.preprocess(plate_text, score)

        return plate_text, score

    def predict(self, frame: np.ndarray, ocr_threshold: float=0.5):
        self.ocr_threshold = ocr_threshold
        plate_text, score = self._predict(frame)

        return plate_text, score


class OCRMotorModel(OCRModel):
    def __init__(self, method_name):
        super().__init__(method_name)

    def preprocess(self, plate_text, score):
        if plate_text == "":
            return
        if score < self.ocr_threshold:
            return
        
        plate_text = ocr_to_motorplate(plate_text)

        return plate_text 

