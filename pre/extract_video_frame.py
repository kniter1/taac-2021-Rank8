import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import time
from torchvision.transforms import transforms
from PIL import Image
import PIL
import argparse
total_frames = 100
img_size = 384
def get_video_frame(video_path, save_path, video_transforms):
    video_list = os.listdir(video_path)
    print(len(video_list))
    for index, video_id in enumerate(video_list):
        cap = cv2.VideoCapture(os.path.join(video_path, video_id))
        fps = int(cap.get(cv2.CAP_PROP_FPS )+ 0.5)
        interval = fps // 1.0
        print('video fps:{}'.format(fps))
        batch = torch.zeros((total_frames, 3, img_size, img_size), dtype=torch.float32)

        counter = 0
        i = 0
        print('start...')
        start = time.time()
        while cap.isOpened():
            if cap.grab():
                ret, frame = cap.retrieve()
            else:
                break

            if not ret:
                break

            if int(counter % interval) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = video_transforms(frame)
                batch[i] = frame
                i += 1

                if i == total_frames:
                    break
            counter += 1
        video_name = video_id.split('.mp4')[0]
        print(batch[:i].shape)
        np.save(os.path.join(save_path, video_name + '.npy'),batch[:i].numpy().astype(np.float16))
        end = time.time()
        print('video{}\{} finished, time:{}'.format(index+1, len(video_list), end-start))


video_path = '/home/tione/notebook/algo-2021/dataset/videos/video_5k/test_5k'
save_path = '/home/tione/notebook/test_5k_384_frame_npy'
video_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
parser = argparse.ArgumentParser(description='extract frame')
parser.add_argument('--video', type=str, default='/home/tione/notebook/algo-2021/dataset/videos/video_5k/test_5k')
parser.add_argument('--save', type=str, default='/home/tione/notebook/test_5k_384_frame_npy')
args = parser.parse_args()
video_path = args.video
save_path = args.save
get_video_frame(video_path, save_path, video_transforms)
