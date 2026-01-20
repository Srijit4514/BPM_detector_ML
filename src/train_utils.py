import torch
import torch.utils.data as data
import cv2
import numpy as np
import os

class RPPGDataset(data.Dataset):
    def __init__(self, video_dir, label_dir, frames=128):
        self.video_dir = video_dir
        self.labels = np.load(os.path.join(label_dir, "ppg.npy"))
        self.videos = sorted(os.listdir(video_dir))
        self.frames = frames

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        vid_path = os.path.join(self.video_dir, self.videos[idx])
        frames = sorted(os.listdir(vid_path))
        frames = frames[:self.frames]

        vid_tensor = []
        for f in frames:
            img = cv2.imread(os.path.join(vid_path, f))
            img = cv2.resize(img, (128, 128))
            vid_tensor.append(img / 255.0)

        vid_tensor = np.stack(vid_tensor)
        vid_tensor = torch.FloatTensor(vid_tensor).permute(0,3,1,2)

        label = torch.FloatTensor(self.labels[idx][:self.frames])
        return vid_tensor, label
