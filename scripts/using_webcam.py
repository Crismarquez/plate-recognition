import cv2
#import matplotlib.pyplot as plt
from recoplate.models import PlateRecognition
from config.config import MODEL_CONFIGURATION, DATA_DIR
from recoplate.utils import draw_first_plate

model_configuration = "motorcycle"
model_configuration = MODEL_CONFIGURATION[model_configuration]
model = PlateRecognition(
    model_configuration["detector_name"],
    model_configuration["ocr_method"]
    )

device = 0
video_path = DATA_DIR / "WIN_20220815_16_13_29_Pro.mp4"
cap = cv2.VideoCapture(device)
#cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"Cannot open {device}")
    exit()

print("Starting to read license plate")
while True:

    ret, frame = cap.read()
    print(frame.shape)
    if not ret:
        print("Cannot receive frames. Exiting...")
        break

    cropped_plate, all_plate_text = model.predict(frame)

    if cropped_plate:
        frame = draw_first_plate(frame, cropped_plate, all_plate_text)

    cv2.imshow("plate-detected", frame)

    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
