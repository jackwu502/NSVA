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
from transformers import AutoFeatureExtractor, AutoImageProcessor, VideoMAEModel, VideoMAEForPreTraining, TimesformerModel, VideoMAEImageProcessor, VideoMAEConfig
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

def get_indices(l, stride=8, batch_size=8):
    num_batches = math.ceil((l / stride) / batch_size)
    padding = num_batches * batch_size - math.ceil(l / stride)
    x = np.arange(0, l, stride)
    x = np.append(x, [0] * padding)
    x = x.reshape((num_batches, batch_size)).astype(np.uint16)
    return x

def include_filter(filename):
    return filename.endswith('.mp4') and not filename.endswith('0021800238-376.mp4')

class VideoDatasetFeatures(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.video_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f)) and include_filter(f)]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        vr = VideoReader(video_path)
        try:
            vr = VideoReader(video_path)
        except:
            return torch.from_numpy(np.zeros([16, 3, 224, 224]))

        # Sample frames
        indices = get_indices(len(vr), batch_size=16)
        frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(3,0,1,2)

        if self.transform:
            frames = self.transform(frames)

        frames = frames.reshape(-1, 16, 3, 224, 224)

        return frames, video_path

class VideoMAEImageProcessorTensor(VideoMAEImageProcessor):
    def preprocess(self, videos, **kwargs):
        processed_videos = [super(VideoMAEImageProcessorTensor, self).preprocess(list(x), **kwargs) for x in videos]
        return processed_videos

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    #model = VideoMAEForPreTraining(VideoMAEConfig())
    #model.load_state_dict(torch.load("/home/ubuntu/lucas/NSVA/finetuned_model_90masking/model_epoch_1.pt"))
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model.to(device)
    image_processor = VideoMAEImageProcessorTensor.from_pretrained("MCG-NJU/videomae-base")

    mean = image_processor.image_mean
    std = image_processor.image_std
    resize_to = image_processor.size['shortest_edge']

    num_frames_to_sample = model.config.num_frames
    sample_rate = 8
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

# Validation and evaluation datasets' transformations.
    val_transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            Resize((resize_to, resize_to)),
        ]
    )

    batch_size = 1  # must be 1 for feature extraction
    num_workers = 8

    train_dataset = VideoDatasetFeatures(root_dir='/home/ubuntu/shared_data/pbp_videos/train/', transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    val_dataset = VideoDatasetFeatures(root_dir='/home/ubuntu/shared_data/pbp_videos/val/', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = VideoDatasetFeatures(root_dir='/home/ubuntu/shared_data/pbp_videos/test/', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    gameid_to_videoid = json.load(open("./tools/gameid_eventid2vid.json"))

    pool_layer = nn.AvgPool1d(1568)

    features = {}

    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
                # Convert the video batch to pixel values and apply normalization
                video_batch, fname = batch
                video_id = fname[0].split('/')[-1].split('.')[0]
                #pixel_values = torch.tensor(np.array([x['pixel_values'] for x in image_processor(video_batch.squeeze(0))])).squeeze(1)
                #print(pixel_values.shape, video_batch.shape)
                pixel_values = video_batch.squeeze(0)
                pixel_values = pixel_values.to(device)

                num_frames = 16
                num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
                seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame

                out = model(pixel_values)
                feats = pool_layer(out.last_hidden_state.permute(0,2,1)).squeeze(2).detach().cpu().numpy()
                features[gameid_to_videoid[video_id]] = feats

            with open('finetuned_model_features.pickle', 'wb') as f:
                pickle.dump(features, f)

if __name__ == "__main__":
    main()
