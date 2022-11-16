import torch
from config.config import DATA_DIR
import cv2

img_path = DATA_DIR / "plate_1.png"
img = cv2.imread(str(img_path))

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.onnx')

results = model(img)  # inference

crops = results.crop(save=False)
print(img.shape)

for crop in crops:
    img_crop = crop["im"]
    cv2.imshow("plate-detected", img_crop)
    cv2.imwrite("output_plate.png", img_crop)
    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows()


