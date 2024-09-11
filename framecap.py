import os
import numpy as np
import cv2
from glob import glob

def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")

def save_frames(video_path, save_dir, gap=10):
    name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    print("before, ", video_path)
    cap = cv2.VideoCapture(video_path)
    print("afterwards")
    idx = 0

    while True:
        ret, frame = cap.read()

        if ret == False:
            cap.release()
            break
        # IF file {save_path}/{idx}.png doesn't exist, then... :
        if not os.path.exists(f"{save_path}/{idx}.png"):
            if idx == 0:
                resized = cv2.resize(frame, (640, 360))
                cv2.imwrite(f"{save_path}/{idx}.png", resized) 
            else:
                if idx % gap == 0:
                    resized = cv2.resize(frame, (640, 360))
                    cv2.imwrite(f"{save_path}/{idx}.png", resized)

        idx += 1

video_paths = "/Users/mdo/Desktop/marioframe/video/MarioKart5.mp4"


save_dir = "save"

save_frames(video_paths, save_dir, gap=20)
