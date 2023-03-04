from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
import torchvision

import torch.nn.functional as F
import pdb
# from visual_utils import Preprocessing, extracting_features, Normalize

class OURDS_Caption_DataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            video_feature,
            bbx_feature,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            split_task = None,
            fine_tune_extractor = False
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))

        self.game_id_to_players = json.load(open('/home/ubuntu/NSVA/tools/video_id_to_players.json', 'r'))
        game_id_to_video_id = json.load(open('/home/ubuntu/NSVA/tools/gameid_eventid2vid.json', 'r'))
        video_id_to_game_id = {}
        for clip_id in game_id_to_video_id.keys():
            game_id = clip_id.split("-")[0]
            video_id_to_game_id[game_id_to_video_id[clip_id]] = game_id
        self.video_id_to_game_id = video_id_to_game_id

        self.fine_tune_extractor = fine_tune_extractor
        if self.fine_tune_extractor == False:
            self.feature_dict = video_feature
            self.bbx_feature_dict = bbx_feature
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.split_task = split_task
        self.videoid2gameid_eventid = json.load(open('videoid2gameid_eventid.json','r'))
        #self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]
        #self.feature_size = 768
        self.feature_size = video_feature['video0'].shape[1]
        self.feature_size_bbx = 768
        self.multibbxs = True
        if self.fine_tune_extractor == True:
            self.video_path = '/home/ubuntu/vcap/content2/pbp_videos'
            # file_weight = "/home/ubuntu/vcap/S3D/S3D_kinetics400.pt"
            # num_class = 400
            # preprocess = Preprocessing("3d")
            # S3D_model = S3D(num_class)
            # # load the weight file and copy the parameters
            # if os.path.isfile(file_weight):
            #     print("loading weight file: %s"%file_weight)
            #     weight_dict = torch.load(file_weight)
            #     model_dict = S3D_model.state_dict()
            #     for name, param in weight_dict.items():
            #         if "module" in name:
            #             name = ".".join(name.split(".")[1:])
            #         if name in model_dict:
            #             if param.size() == model_dict[name].size():
            #                 model_dict[name].copy_(param)
            #             else:
            #                 print(" size? " + name, param.size(), model_dict[name].size())
            #         else:
            #             print(" name? " + name)

            #     print(" loaded")
            # else:
            #     print("weight file?")
            # S3D_model = S3D_model.cuda()
            # self.S3D_model = S3D_model
            # self.S3D_model.train()
        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        #video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        #video_ids = ['video%s'%i for i in range(len(self.feature_dict))]

        # if self.fine_tune_extractor == True:
        #     del self.feature_dict

        prefixes = ['<T1>_','<T2>_','<T3>_','<T4>_']
        prefixes_in_use = [prefixes[s_id] for s_id,s_t in enumerate(self.split_task) if s_t ==1]
        sentences_in_use = []
        for sen in self.data['sentences']:
            use_sentence = False
            for p_i_u in prefixes_in_use:
                if p_i_u in sen['caption']:
                    use_sentence = True
                    break

            if use_sentence:
                sentences_in_use.append(sen)

        #self.data['sentences'] = sentences_in_use
        # only for temporary use, need to split dataset by game instead of video clip
        # train_nn = 35721
        # valid_nn = 4465
        # test_nn = 4465
        split_dict = json.load(open('split2video_id_after_videos_combination.json', 'r'))
        self.videoid2split = {y:x for x in split_dict.keys() for y in split_dict[x]}
        #split_dict = {"train": video_ids[:train_nn], "val": video_ids[train_nn:train_nn + valid_nn], "test": video_ids[train_nn+valid_nn:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    #self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            all_train = [id for id, task in enumerate(self.split_task) if task ==1] 

            for a_t in all_train:
                for vidx,vid in enumerate(choiced_video_ids):
                #self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
                    self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][a_t])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    # if itm['video_id'] == 'video34280':
                    #     print()
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])

            random_val = [id for id, task in enumerate(self.split_task) if task ==1] 

            for vidx,vid in enumerate(choiced_video_ids):
                #self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][random_val[vidx%len(random_val)]])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            task_type = -1
            if caption is not None:
                if '<T1>_' in caption and self.split_task[0] ==1:
                    task_type = 0
                    caption = caption[5:]
                elif '<T2>_' in caption and self.split_task[1] ==1:
                    task_type = 1
                    caption = caption[5:]
                elif '<T3>_' in caption and self.split_task[2] ==1:
                    task_type = 2
                    caption = caption[5:]
                elif '<T4>_' in caption and self.split_task[3] ==1:
                    task_type = 3
                    caption = caption[5:]
                else:
                    assert task_type!=-1
                caption_words = self.tokenizer.tokenize(caption)
            else:
                assert 1==0
                caption_words = self._get_single_text(video_id)
            
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)
        assert task_type!=-1
        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids, task_type

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):

            if self.fine_tune_extractor == False:
                video_slice = self.feature_dict[video_id]
            # else:
            #     gameid_eventid = self.videoid2gameid_eventid[video_id]
            #     # gameid = gameid_eventid.split('-')[0]
            #     # eventid = gameid_eventid.split('-')[1]
            #     video_slice = extracting_features(self.video_path+'/'+self.videoid2split[video_id]+'/'+gameid_eventid+'.mp4')

            # print("########{}".format(video_slice.shape))
            # if self.feature_size == 1024:
            #     video_slice = np.transpose(video_slice)
            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                # print("!!!!!!!!!!! {}, {}".format(video[i][:slice_shape[0]].shape, video_slice.shape))
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def _get_bbx(self, choice_video_ids):
        ## feature size for BBX is  
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size_bbx), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):

            if self.fine_tune_extractor == False:
                if video_id not in self.bbx_feature_dict.keys():
                    video_slice = video
                    return video, video_mask, None, None
                else:
                    video_slice = self.bbx_feature_dict[video_id]
            # else:
            #     gameid_eventid = self.videoid2gameid_eventid[video_id]
            #     # gameid = gameid_eventid.split('-')[0]
            #     # eventid = gameid_eventid.split('-')[1]
            #     video_slice = extracting_features(self.video_path+'/'+self.videoid2split[video_id]+'/'+gameid_eventid+'.mp4')

            # print("########{}".format(video_slice.shape))
            # if self.feature_size == 1024:
            #     video_slice = np.transpose(video_slice)
            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                # print("!!!!!!!!!!! {}, {}".format(video[i][:slice_shape[0]].shape, video_slice.shape))
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index
        #(1,30,768),(1*30),(1,30,768),(1*30)
    def _get_multi_bbxs(self, choice_video_ids):
        ## feature size for BBX is
        # if len(self.bbx_feature_dict['video10981'].shape) ==2:
        #     video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        # else:
        #     video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids),self.bbx_feature_dict['video11030'].shape[0], self.max_frames, self.feature_size_bbx), dtype=np.float32)
        for i, video_id in enumerate(choice_video_ids):

            if self.fine_tune_extractor == False:
                if video_id not in self.bbx_feature_dict.keys():
                    video_slice = video
                    return video, video_mask, None, None
                else:
                    video_slice = self.bbx_feature_dict[video_id]
            # else:
            #     gameid_eventid = self.videoid2gameid_eventid[video_id]
            #     # gameid = gameid_eventid.split('-')[0]
            #     # eventid = gameid_eventid.split('-')[1]
            #     video_slice = extracting_features(self.video_path+'/'+self.videoid2split[video_id]+'/'+gameid_eventid+'.mp4')

            # print("########{}".format(video_slice.shape))
            # if self.feature_size == 1024:
            #     video_slice = np.transpose(video_slice)
            if self.max_frames < video_slice.shape[1]:
                video_slice = video_slice[:,:self.max_frames,:]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[1] else slice_shape[1]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                # print("!!!!!!!!!!! {}, {}".format(video[i][:slice_shape[0]].shape, video_slice.shape))
                for j in range(slice_shape[0]):
                    video[i][j][:slice_shape[1]] = video_slice[j]

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_[0]):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        for k in range(video_pair_.shape[0]):
                            masked_video[i][k][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        game_id = self.video_id_to_game_id[video_id]
        player_ids = self.game_id_to_players[game_id]

        # Find possible player vocab indices
        eligible_player_indices = []
        keys = self.tokenizer.vocab.keys()
        for i in range(len(player_ids)):
            token = "player" + str(player_ids[i])
            if token in keys:
                eligible_player_indices.append(self.tokenizer.vocab[token])

        blocked_player_indices = np.zeros(len(self.tokenizer.vocab))
        for i in range(105, 289):
            if i not in eligible_player_indices:
                blocked_player_indices[i] = 1
        player_ids = blocked_player_indices

        video, video_mask, masked_video, video_labels_index = self._get_video([video_id])
        
        #bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_bbx([video_id])

        if self.multibbxs:
            bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_multi_bbxs([video_id])

        else:
            bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_bbx([video_id])

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids, task_type = self._get_text(video_id, caption)

        #video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, bbx, bbx_mask, player_ids

