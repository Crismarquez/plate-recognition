from time import sleep

import cv2
import click
import pyperclip

from config.config import MODEL_CONFIGURATION
from recoplate.models import PlateRecognition, PlateRecognitionLite
from recoplate.utils import draw_first_plate


@click.group()
def cli():
    pass

@cli.command()  
@click.argument("model_configuration", default="motorcycle")
@click.argument("device", default=0)
def webcam(device, model_configuration):

    model_configuration = MODEL_CONFIGURATION[model_configuration]
    model = PlateRecognition(
        model_configuration["detector_name"],
        model_configuration["ocr_method"]
        )

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"Cannot open {device=}")
        exit()
    
    print("Starting to read license plates")
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frames. Exiting...")
            break

        cropped_plate, all_plate_text = model.predict(frame)

        for plate_detected, text in zip(cropped_plate, all_plate_text):
            print(text)

    cap.release()
    cv2.destroyAllWindows()

    ...

@cli.command()  
@click.argument("model_configuration", default="motorcycle")
@click.argument("device", default=0)
def interactive(device, model_configuration):

    model_configuration = MODEL_CONFIGURATION[model_configuration]
    model = PlateRecognition(
        model_configuration["detector_name"],
        model_configuration["ocr_method"]
        )

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"Cannot open {device=}")
        exit()
    
    print("Starting to read license plates")
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frames. Exiting...")
            break

        cropped_plate, all_plate_text = model.predict(frame)

        if cropped_plate:
            frame = draw_first_plate(frame, cropped_plate, all_plate_text)
            if all_plate_text[0]:
                pyperclip.copy(all_plate_text[0])

        cv2.imshow("plate-detected", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    ...

@cli.command()  
@click.argument("model_configuration", default="motorcycle")
@click.argument("device", default=0)
def interactivelite(device, model_configuration):

    model_configuration = MODEL_CONFIGURATION[model_configuration]
    model = PlateRecognitionLite(
        model_configuration["ocr_method"]
        )

    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print(f"Cannot open {device=}")
        exit()
    
    total_frames = 0
    skip_frames = 10

    size_plate = [200, 300]
    print("Starting to read license plates")
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive frames. Exiting...")
            break
        
        if total_frames % skip_frames:
            cropped_plate, all_plate_text = model.predict(frame, size_plate)

            frame = draw_first_plate(frame, cropped_plate, all_plate_text, size_plate)
            if all_plate_text[0]:
                pyperclip.copy(all_plate_text[0])

        total_frames += 1

        if total_frames > 10000:
            total_frames = 0

        cv2.imshow("plate-detected", frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()

    ...

if __name__ == "__main__":
    cli()
