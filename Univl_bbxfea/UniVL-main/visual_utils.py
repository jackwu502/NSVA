import os
import numpy as np
import cv2
# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
from s3d_model import S3D
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import math
import timeit
import torch
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-worker", type=int,
                    help="worker#")

# """ Output the top 5 Kinetics classes predicted by the model """
file_weight = "/home/ubuntu/vcap/S3D/S3D_kinetics400.pt"
# file_weight = "/home/ubuntu/vcap/content2/MIL-NCE_HowTo100M/checkpoint/milnce/epoch0006.pth.tar"
num_class = 400

import torch as th


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = th.FloatTensor(mean).view(1, 3, 1, 1)
        self.std = th.FloatTensor(std).view(1, 3, 1, 1)

    def __call__(self, tensor):
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        return tensor


### Borrow pre-processor from
class Preprocessing(object):
    def __init__(self, type):
        self.type = type
        if type == "2d":
            self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif type == "3d":
            self.norm = Normalize(mean=[110.6, 103.2, 96.3], std=[1.0, 1.0, 1.0])

    def _zero_pad(self, tensor, size):
        n = size - len(tensor) % size
        if n == size:
            return tensor
        else:
            z = th.zeros(n, tensor.shape[1], tensor.shape[2], tensor.shape[3])
            return th.cat((tensor, z), 0)

    def __call__(self, tensor):
        if self.type == "2d":
            tensor = tensor / 255.0
            tensor = self.norm(tensor)
        elif self.type == "3d":
            # tensor = self._zero_pad(tensor, 16)
            tensor = self.norm(tensor)
            # tensor = tensor.view(-1, 16, 3, 112, 112)
            # tensor = tensor.transpose(1, 2)
        return tensor


preprocess = Preprocessing("3d")
model = S3D(num_class)


# load the weight file and copy the parameters
if os.path.isfile(file_weight):
    print("loading weight file: %s"%file_weight)
    weight_dict = th.load(file_weight)
    model_dict = model.state_dict()
    for name, param in weight_dict.items():
        if "module" in name:
            name = ".".join(name.split(".")[1:])
        if name in model_dict:
            if param.size() == model_dict[name].size():
                model_dict[name].copy_(param)
            else:
                print(" size? " + name, param.size(), model_dict[name].size())
        else:
            print(" name? " + name)

    print(" loaded")
else:
    print("weight file?")

model = model.cuda()
th.backends.cudnn.benchmark = False
model.eval()
# model.train()
# fc1 = nn.Linear(24, 16)
# fc1.cuda()
def extracting_features(mp4_path):


    """mp4 io to torch tensor """
    vid_f, vid_a, vid_meta = torchvision.io.read_video(
        mp4_path, pts_unit="sec"
    )

    """do not ignore """
    # if vid_f.shape[0] < 64:
    # print("{} is too short".format(mp4))
    # continue

    clip = F.interpolate(vid_f.permute(0, 3, 1, 2), size=(256, 384))
    clip = clip.unsqueeze(0).permute(0, 2, 1, 3, 4).float()

    input_temporal = 15
    num_frame = clip.shape[2]
    num_sec = math.ceil(num_frame * 1.0 / int(vid_meta["video_fps"]))
    downsample_frames = math.ceil(
        num_frame / int(vid_meta["video_fps"]) * input_temporal
    )

    # downsample_frames = 8
    """Down-Sample to 8 frames per second, rather than 60 frames """
    num_idx = np.round(np.linspace(0, num_frame - 1, downsample_frames)).astype(int)

    """Comment out if not do any downsampling """
    #clip = clip[:, :, num_idx]

    with th.no_grad():
        """Output is [1, fea_dim, T/8, 1, 1]"""
        clip = preprocess(clip.squeeze().permute(1, 0, 2, 3))
        clip.cuda()
        num_iter = math.ceil(clip.shape[0] / 64)
        rst_list = []
        for iter in range(num_iter):
            min_ind = iter * 64
            max_ind = (iter + 1) * 64

            if iter == num_iter - 1:
                rst_list.append(
                    model.fea_extract(
                        clip[-64:].unsqueeze(0).permute(0, 2, 1, 3, 4).cuda()
                    )
                )
            else:

                rst_list.append(
                    model.fea_extract(
                        # model(
                        clip[min_ind:max_ind]
                        .unsqueeze(0)
                        .permute(0, 2, 1, 3, 4)
                        .cuda()
                    )
                )


        vid_fea = th.cat(rst_list, 2)
        vid_fea = th.squeeze(vid_fea)
        downsample_frames = 16
        num_idx = np.round(np.linspace(0, vid_fea.shape[1] - 1, downsample_frames)).astype(int)
        vid_fea = vid_fea[:, num_idx]

        # assert vid_fea.size(1) <= 80
        # pad_vid_fea = torch.zeros((vid_fea.size(0),80), device=vid_fea.device, dtype=vid_fea.dtype)
        # pad_vid_fea[:,:vid_fea.size(1)] = vid_fea
        # pad_vid_fea = fc1(pad_vid_fea)
    return vid_fea.t().detach().cpu().numpy()
        # with open(os.path.join(out_path, mp4.replace(".mp4", ".npy")), "+wb") as f:
        #     np.save(f, np.squeeze(vid_fea))  

# def main(file_path,mp42mp4_path_out_path):
#     global model

#     """Read mp4 files"""
#     mp4_path = os.path.join("/home/ubuntu/vcap/content2/pbp_videos", file_path)
#     out_path = os.path.join("/home/ubuntu/vcap/S3D/output_features", file_path)

#     if not os.path.exists(out_path):
#         os.mkdir(out_path)
#     all_mp4 = os.listdir(mp4_path)
#     all_mp4.sort()
#     # sublist_size = int(np.ceil(len(all_mp4) / 2))
#     # lst_chucks = chunks(all_mp4,sublist_size)
#     # parameters_list = []
#     # for l_c in lst_chucks:
#     for mp4 in all_mp4:
#         mp42mp4_path_out_path[mp4] = mp4_path+'<sep>'+out_path
#         # try:
#         #     _thread.start_new_thread(extracting_features,(l_c,mp4_path,out_path,model))
#         # except BaseException as e:
#         #     print(e)

#     return mp42mp4_path_out_path


def transform(snippet):
    """ stack & noralization """
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.0).sub_(255).div(255)

    return snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(
        0, 2, 1, 3, 4
    )
def findFiles(path): return glob.glob(path)

if __name__ == "__main__":
    # vid_folders = os.listdir("/home/ubuntu/vcap/content2/pbp_videos/test")
    test_videos = list(set(findFiles('/home/ubuntu/vcap/content2/pbp_videos/test/*')))
    extracting_features(test_videos[0])
    print()
    
    # args = parser.parse_args()

    # mp42mp4_path_out_path = {}
    # for f in vid_folders:

    #     mp42mp4_path_out_path = main(f,mp42mp4_path_out_path)

    # all_mp4 = list(mp42mp4_path_out_path.keys())
    # all_mp4.sort()
    # print(len(all_mp4))
    # sublist_size = int(np.ceil(len(all_mp4) / 3))
    # lst_chucks = chunks(all_mp4,sublist_size)
    # subset_mp4_for_worker = lst_chucks[args.worker]
    # for subset_mp4 in subset_mp4_for_worker:
    #     v = mp42mp4_path_out_path[subset_mp4]
    #     extracting_features([subset_mp4],v.split('<sep>')[0],v.split('<sep>')[1],model)
    # print()


