import os
import subprocess
from ultralytics import YOLO


# def detect_objects(input_video, output_video, weights='best.pt', img_size=640, conf=0.6):
#     command = f'python "yolov5/detect.py" --weights {weights} --img {img_size} --conf {conf} --source {input_video} --project static/processed --name "" --save-txt --exist-ok'
#     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
#     return result.stdout
# def detect_objects(input_video, output_video, weights='best.pt', img_size=640, conf=0.6):
    # command = f'python "ultralytics/detect.py" --weights {weights} --img {img_size} --conf {conf} --source {input_video} --project static/processed --name "" --save-txt --exist-ok'
    # result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    # return result.stdout

# model.predict(source="", show=True, save=True, hide_labels=False, hide_conf=False, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)


def detect_objects(input_video, output_video):
    model = YOLO("best.pt")

    output_directory = "static"  # Change this to your desired output directory

    model.predict(
        source=input_video,
        show=False,
        save=True,
        hide_labels=False,
        hide_conf=False,
        conf=0.45,
        save_txt=False,
        save_crop=False,
        line_thickness=2,
        project=output_directory,
        exist_ok=True
    )