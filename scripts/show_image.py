from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from config.config import DATA_DIR
from recoplate.models import PlateRecognition
from recoplate.utils import resize_img

img_path = DATA_DIR / "plate_4.png"
img = cv2.imread(str(img_path))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
model = PlateRecognition("mobilenet_v2", "paddle")

cropped_plate, all_plate_text = model.predict(img)

x = int(img.shape[0] * 0.6)
y = int(img.shape[1] * 0.1)

# size for show plate
width = int(img.shape[0] * 0.35)
height = int(img.shape[1] * 0.15)

for plate, plate_text in zip(cropped_plate, all_plate_text):
    
    plate = resize_img(plate, width, height)
    cv2.putText(img, plate_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 10, 58), 3)
    img[y+10:y+10+height, x:x+width] = plate
    # plt.imshow(img)
    # plt.show()
    cv2.imshow("plate-detected", img)
    cv2.imwrite("output_plate.png", img)
    cv2.waitKey(0) 
    #closing all open windows 
    cv2.destroyAllWindows()
