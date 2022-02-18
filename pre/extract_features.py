from VIT.src.model import *
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import argparse
import time
total_frames = 100
img_size = 384
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(image_size=(384, 384),
                 patch_size=(16, 16),
                 emb_dim=1024,
                 mlp_dim=4096,
                 num_heads=16,
                 num_layers=24,
                 num_classes=1000,
                 attn_dropout_rate=0.0,
                 dropout_rate=0.1,
                 feat_dim=None)
state_dict = torch.load('/home/tione/notebook/taac-2021-S/pretrained_models/VIT-pretrained_models/imagenet21k+imagenet2012_ViT-L_16.pth', map_location='cpu')
model.load_state_dict(state_dict['state_dict'])
model = model.to(device)
class VideoDataset(Dataset):
    def __init__(self, video_npy_path):
        super().__init__()
        self.npy_path = video_npy_path
        self.video_list = os.listdir(video_npy_path)
   

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        video_name = self.video_list[index]
        frames = np.load(os.path.join(self.npy_path, video_name)).astype(np.float32)

        return frames

  
def extract_features(video_frames_path, features_save_path):
    video_npy_list = os.listdir(video_frames_path)
    print('***** data loading *****')
    dataset = VideoDataset(
        video_npy_path=video_frames_path
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=12
    )
    print('num example:{}'.format(len(dataset)))
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            batch = batch.to(device)
            frames = batch[0]
            preds = model(frames)
            print('features shape{}'.format(preds.shape))
            data = preds.data.cpu().numpy()
            video_name = video_npy_list[index].split('.npy')[0]
            np.save(os.path.join(features_save_path, video_name+'.npy'), data)
            print('feature extract finished:{}\{}'.format(index+1, len(dataloader)))
    print('***** finfished *****')
parser = argparse.ArgumentParser(description='extract features')
parser.add_argument('--frame', type=str, default='/home/tione/notebook/test_5k_384_frame_npy')
parser.add_argument('--save', type=str, default='/home/tione/notebook/VIT_L_test_5k_features')
args = parser.parse_args()
video_frames_path = args.frame
features_save_path = args.save
start = time.time()
extract_features(video_frames_path, features_save_path)
end = time.time()
print('total time:{}'.format(end-start))