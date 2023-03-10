import os
import sys
import pickle
import json
import numpy as np
import math
import av
import torch
import torch.nn as nn
import cv2
import tqdm
import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import TrainingArguments, Trainer
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoImageProcessor, VideoMAEModel, VideoMAEForPreTraining, TimesformerModel, VideoMAEImageProcessor
from decord import VideoReader, cpu
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def include_filter(filename):
    return filename.endswith('.mp4') and not filename.endswith('0021800238-376.mp4')

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.video_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f)) and include_filter(f)]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        try:
            vr = VideoReader(video_path)
        except:
            return torch.from_numpy(np.zeros([16, 3, 224, 224]))

        # Sample frames
        indices = sample_frame_indices(clip_len=16, frame_sample_rate=8, seg_len=len(vr))
        frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(3,0,1,2)

        if self.transform:
            frames = self.transform(frames)

        frames = frames.permute(1,0,2,3)

        return frames

class VideoMAEImageProcessorTensor(VideoMAEImageProcessor):
    def preprocess(self, videos, **kwargs):
        processed_videos = [super(VideoMAEImageProcessorTensor, self).preprocess(list(x), **kwargs) for x in videos]
        return processed_videos

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 1
    num_workers = 8
    num_epochs = 2

    # Initialize the model and image processor
    model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    image_processor = VideoMAEImageProcessorTensor.from_pretrained("MCG-NJU/videomae-base")

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    mean = image_processor.image_mean
    std = image_processor.image_std
    resize_to = image_processor.size['shortest_edge']

    num_frames_to_sample = model.config.num_frames
    sample_rate = 8
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    # Training dataset transformations.
    train_transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            RandomShortSideScale(min_size=256, max_size=320),
            RandomCrop(resize_to),
            RandomHorizontalFlip(p=0.5),
        ]
    )

    # Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            Resize((resize_to, resize_to)),
        ]
    )

    base_transform = Compose([
        ToTensor(),
    ])

    train_dataset = VideoDataset(root_dir='/home/ubuntu/shared_data/pbp_videos/train/', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = VideoDataset(root_dir='/home/ubuntu/shared_data/pbp_videos/val/', transform=val_transform)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = VideoDataset(root_dir='/home/ubuntu/shared_data/pbp_videos/test/', transform=val_transform)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    train_loss_list = []

    # Loop through each epoch
    for epoch in range(num_epochs):
        # Loop through each batch
        for batch_idx, video_batch in enumerate(tqdm.tqdm(train_loader)):
            # Convert the video batch to pixel values and apply normalization
            pixel_values = torch.from_numpy(np.array([x['pixel_values'] for x in image_processor(video_batch)])).squeeze(1)
            pixel_values = pixel_values.to(device)

            # Feed the pixel values through the model
            num_frames = 16
            num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
            seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

            # bool_masked_pos_list = [torch.randint(0, 2, (1, seq_length)).bool() for _ in range(batch_size)]
            # bool_masked_pos = torch.cat(bool_masked_pos_list)
            # bool_masked_pos = bool_masked_pos.to(device)

            mask_ratio = 0.9
            bool_masked_pos = np.ones(seq_length)
            mask_num = math.ceil(seq_length * mask_ratio)
            mask = np.random.choice(seq_length, mask_num, replace=False)
            bool_masked_pos[mask] = 0
            bool_masked_pos = torch.as_tensor(bool_masked_pos).bool().unsqueeze(0)
            bool_masked_pos = torch.cat([bool_masked_pos for _ in range(batch_size)])

            outputs = model(pixel_values, bool_masked_pos)

            # Calculate the loss
            loss = outputs.loss

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the model every epoch
            if batch_idx == len(train_loader) - 1:
                torch.save(model.state_dict(), f"finetuned_model/model_epoch_{epoch}.pt")

            # Save the training loss every batch
            if batch_idx % 1 == 0:
                train_loss_list.append((epoch, batch_idx, loss.item()))
                np.savetxt("finetuned_model/train_loss.txt", train_loss_list)

            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item()}")

if __name__ == "__main__":
    main()

