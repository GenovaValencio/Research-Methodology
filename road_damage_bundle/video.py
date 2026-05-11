import cv2
import os

image_folder = 'data'
video_name = 'road_damage_video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

video.release()