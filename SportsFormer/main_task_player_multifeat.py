from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from distutils.log import debug
from posixpath import split
from sys import breakpointhook

import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from collections import OrderedDict
from nlgeval import NLGEval
import time
import argparse
import json
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_multifeat import UniVL
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption import MSRVTT_Caption_DataLoader
from dataloaders.dataloader_ourds_caption_multifeat import OURDS_Caption_DataLoader
from eval_utils import *
from util import *
from torch import nn
from torchsummary import summary
import pickle5 as pickle
import re
torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='UniVL on Caption Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--use_prefix_tuning", action='store_true', help="Whether to use prefix tuning.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='./data/ourds_train.44k.csv', help='')
    parser.add_argument('--val_csv', type=str, default='./data/ourds_JSFUSION_test.csv', help='')
    parser.add_argument('--data_path', type=str, default='./data/ourds_description.json',
                        help='caption and transcription pickle file path')
    
    '''Initialize multi-modal features from here:
    feature_path (mandatory):   [Timesformer or S3D] video features,        shape [numFrames, dimFeature (768)]
    allplayer_feature_path:     [Timesformer] on cropped players,           shape [numFrames, numPlayers, dimFeature (768)]
    kp_features_path:           [Timesformer] on plotted body key-points,   shape [numFrames, dimFeature (768)]
    court_features_path:        [Timesformer] on plotted courtling seg,     shape [numFrames, dimFeature (768 * 2)]
    bbx_features_path:          [Timesformer] on summed cls2+ball+basket,   shape [numFrames, dimFeature (768)]
    '''
    parser.add_argument('--features_path', type=str, default='/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_videos_timesformer_features.pickle',
                        help='feature path for 2D features')
    # parser.add_argument('--courtseg_features_path', type=str, default='/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_courtlineseg_data/ourds_videos_features.pickle',
    #                     help='feature path for 2D features')
    # parser.add_argument('--bbxcls2_features_path', type=str, default='/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_cls2_data/ourds_videos_features.pickle',
    #                     help='feature path for 2D features')
    # parser.add_argument('--bbxball_features_path', type=str, default='/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_ball_data/ourds_videos_features.pickle',
    #                     help='feature path for 2D features')
    # parser.add_argument('--bbxbasket_features_path', type=str, default='/local/riemann1/home/zhufl/hdd1/UniVL_processing_code/ourds_basket_data/ourds_videos_features.pickle',
    #                     help='feature path for 2D features')

    "Use re-organzied feature from JackWu"
    parser.add_argument('--cls2_ball_basket_sum_concat_courtseg_path', type=str, default='./data/cls2_ball_basket_sum_concat_original_courtline_fea.pickle',
                        help='feature path for 2D features')

    parser.add_argument('--num_thread_reader', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=3e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=4, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=300, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=768, help='video feature dimension') # switch between 768 and 1024?

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=30, help='')
    parser.add_argument('--max_frames', type=int, default=30, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    # parser.add_argument("--output_dir", default='/media/chris/hdd1/UniVL_processing_code/ourds_data/ckpt_ourds_caption', type=str, required=False,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument("--output_dir", default='/media/chris/hdd1/UniVL_processing_code/multifea_action_recongition/ckpt_ourds_caption', type=str, required=False,
                        # help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir", default='./output/ckpt_ourds_caption_actionFine', type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=False, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    # parser.add_argument("--init_model", default='/media/chris/hdd1/UniVL_processing_code/UniVL-main/weight/univl.pretrained.bin', type=str, required=False, help="Initial model.")
    parser.add_argument("--init_model", default='./weight/univl.pretrained.bin', type=str, required=False, help="Initial model.")

    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument("--datatype", default="ourds", type=str, help="Point the dataset `youcook` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=2, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=6, help="Layer NO. of decoder.")

    '''
    T1: Player Recognition
    T2: Action Prediction (Recognition)
    T3: Description Generation
    T4: Commentary Generation
    '''

    parser.add_argument('--train_tasks', default=[1,0,0,0],type=lambda s: [int(item) for item in s.split(',')], help="train with specific tasks: 1 for yes, 0 for no")
    parser.add_argument('--test_tasks',default=[1,0,0,0], type=lambda s: [int(item) for item in s.split(',')], help="test with specific tasks: 1 for yes, 0 for no")
    parser.add_argument('--t1_postprocessing', action='store_true', help="Whether postprocess output with action type")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
    parser.add_argument('--action_level', default=1, help="Whether decide which action do we want to perform recognition, range from 0-2")
    args = parser.parse_args()

    args.do_train = True
    args.stage_two = True
    args.do_lower_case = True
    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args
def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank, type_vocab_size=2):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, type_vocab_size=type_vocab_size)
    # model = model.float()
    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def dataloader_youcook_train(args, tokenizer):
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(youcook_dataset), train_sampler

def dataloader_youcook_test(args, tokenizer):
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_ourds_train(args, tokenizer):
    ourds_dataset = OURDS_Caption_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        feature_tuples=args.feature_tuple,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
        split_task = args.train_tasks,
        gameid2videoid='./data/gameid_eventid2vid.json',
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(ourds_dataset)
    dataloader = DataLoader(
        ourds_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(ourds_dataset), train_sampler

def dataloader_ourds_test(args, tokenizer, split_type="test"):
    ourds_testset = OURDS_Caption_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        video_feature=args.video_feature,
        feature_tuples=args.feature_tuple,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        split_task = args.test_tasks,
        gameid2videoid='./data/gameid_eventid2vid.json',
    )

    test_sampler = SequentialSampler(ourds_testset)
    dataloader_ourds = DataLoader(
        ourds_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        # drop_last=False,
        drop_last=True,
    )
    return dataloader_ourds, len(ourds_testset)

def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))

    #model_file = '/home/ubuntu/vcap/content/ckpts/ckpt_ourds_caption/pytorch_model.bin.6'
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = UniVL.from_pretrained(args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):

        feature_tuple, feature_mask_tuple = batch[-2:]
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch[:-2])
        # print("One batch data takes {} to prepare".format(time2-time1))

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type = batch

        for key, value in feature_tuple.items():
            feature_tuple[key] = value.to(device=device, non_blocking=True)
        for key, value in feature_mask_tuple.items():
            feature_mask_tuple[key] = value.to(device=device, non_blocking=True)
        
        time1 = time.time()
        loss = model(input_ids, segment_ids, input_mask, video.float(), video_mask.float(),
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                     output_caption_ids=pairs_output_caption_ids,task_type=task_type, feature_tuple=feature_tuple, feature_tuple_mask=feature_mask_tuple)
        time2 = time.time()
        # print("One batch captioning result takes {} to run".format(time2-time1))

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

# ---------------------------------------->
def get_inst_idx_to_tensor_position_map(inst_idx_list):
    ''' Indicate the position of an instance in a tensor. '''
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}


def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    ''' Collect tensor parts associated to active instances. '''

    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)

    return beamed_tensor


def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, feature_tuple, feature_mask_tuple, task_type_rpt = input_tuples

    # Sentences which are still active are collected,
    # so the decoder will not run on completed sentences.
    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_sequence_output_rpt = collect_active_part(sequence_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    new_feature_tuple = {}
    for key, value in feature_tuple.items():
        new_feature_tuple[key] = collect_active_part(value, active_inst_idx, n_prev_active_inst, n_bm)

    active_input_ids_rpt = collect_active_part(input_ids_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_input_mask_rpt = collect_active_part(input_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    new_feature_mask_tuple = {}
    for key, value in feature_mask_tuple.items():
        new_feature_mask_tuple[key] = collect_active_part(value, active_inst_idx, n_prev_active_inst, n_bm)

    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
    active_task_type_rpt = collect_active_part(task_type_rpt, active_inst_idx, n_prev_active_inst, n_bm)

    return (active_sequence_output_rpt, active_visual_output_rpt, active_input_ids_rpt, active_input_mask_rpt, active_video_mask_rpt, new_feature_tuple, new_feature_mask_tuple, active_task_type_rpt), \
           active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None,task_type = None):

    assert isinstance(input_tuples, tuple)

    ''' Decode and update beam status, and then return active beam idx'''
    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples,task_type = task_type):
        sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, feature_tuple, feature_mask_tuple = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)

        # assert court_output_rpt.shape == bbz_output_rpt.shape, "Wrong shape court {} vs. bbx {}".format(court_output_rpt.shape, bbz_output_rpt.shape)
        # assert court_output_rpt.shape == visual_output_rpt.shape, "Wrong shape court {} vs. feat {}".format(court_output_rpt.shape, visual_output_rpt.shape)
        # assert bbz_output_rpt.shape == visual_output_rpt.shape, "Wrong shape bbx {} vs. feat {}".format(bbz_output_rpt.shape, visual_output_rpt.shape)

        dec_output = decoder(sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt,
                             video_mask_rpt, next_decoder_ids, next_decoder_mask, feature_tuple, feature_mask_tuple, shaped=True, get_logits=True,task_type = task_type)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]

        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)

    # Update the beam with predicted word prob information and collect incomplete instances
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map,
                                                        decoder_length=decoder_length)

    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]

        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores
# >----------------------------------------

def eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    test_tasks = [id for id, task in enumerate(args.test_tasks) if task ==1] 
    result_list_byTask = {t:[] for t in test_tasks}
    caption_list_byTask = {t:[] for t in test_tasks}
    all_result_lists = []
    all_rst_lists = []
    all_gt_lists = []
    all_caption_lists = []
    model.eval()

    for b_id, batch in enumerate(test_dataloader):
        # if b_id > 1:
        #     continue

        feature_tuple, feature_mask_tuple = batch[-2:]
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch[:-2])

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type = batch

        "Map feature tuple to cuda()"
        for key, value in feature_tuple.items():
            feature_tuple[key] = value.to(device=device, non_blocking=True)
        for key, value in feature_mask_tuple.items():
            feature_mask_tuple[key] = value.to(device=device, non_blocking=True)

        with torch.no_grad():
            sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask,task_type = task_type)
            fea_output_tuple = {}
            fea_output_mask_tuple = {}
            for (name1, fea), (name2, mask) in zip(feature_tuple.items(), feature_mask_tuple.items()):
                if name1 == 'allplayer':
                    "First, collapse numFrames to batch"
                    fea = fea.view(-1, 10, 768) # [batch * numFrame, 10, 768]
                    mask = mask.view(-1, 10)
                    fea = model.get_fea_output(fea, mask, name1).sum(-2)

                    "Then, recover it and run again on seq dimension"
                    fea = fea.reshape(-1, args.max_frames, 768)
                    mask = mask.reshape(-1, args.max_frames, 10).sum(-1)
                    mask = torch.where(mask !=0, 1, 0)
                    fea = model.get_fea_output(fea, mask, name1, True) # set to True to use another encoder for all-players
                    fea_output_tuple[name1] = fea
                    fea_output_mask_tuple[name1] = mask
                else:
                    fea_output_tuple[name1] = model.get_fea_output(fea, mask, name1)
                    fea_output_mask_tuple[name1] = mask

            feature_mask_tuple = fea_output_mask_tuple

            # -- Repeat data for beam search
            n_bm = 5 # beam_size
            device = sequence_output.device
            n_inst, len_s, d_h = sequence_output.size()
            _, len_v, v_h = visual_output.size()

            decoder = model.decoder_caption

            # Note: shaped first, then decoder need the parameter shaped=True
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            input_mask = input_mask.view(-1, input_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            
            for key, value in feature_mask_tuple.items():
                value = value.view(-1, value.shape[-1])
                feature_mask_tuple[key] = value
            # bbx_mask = bbx_mask.view(-1, bbx_mask.shape[-1])
            # court_mask = court_mask.view(-1, court_mask.shape[-1])

            # The following line need to be changed soon
            if args.use_prefix_tuning !=False:
                video_mask = torch.cat([torch.zeros(video_mask.size(0),model.visual_config.preseqlens[0],dtype=video_mask.dtype,device=video_mask.device),video_mask],dim=1)

            sequence_output_rpt = sequence_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)
            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)

            new_feat_output_tuple = {}
            for key, value in fea_output_tuple.items():
                value = value.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
                new_feat_output_tuple[key] = value

            input_ids_rpt = input_ids.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            input_mask_rpt = input_mask.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            # bbx_mask_rpt = bbx_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)
            # court_mask_rpt = court_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            new_feature_mask_tuple = {}
            for key, value in feature_mask_tuple.items():
                value = value.repeat(1, n_bm).view(n_inst * n_bm, len_v)
                new_feature_mask_tuple[key] = value

            task_type_rpt = task_type.repeat(n_bm)

            # -- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            # -- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            # {0:0, 1:1, 2:2, 3:3, 4:4}
            # print(inst_idx_to_position_map)
            # print(list(range(n_inst)))
            # breakpoint()

            # -- Decode
            for len_dec_seq in range(1, args.max_words + 1):

                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, new_feat_output_tuple, new_feature_mask_tuple), task_type = task_type_rpt)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                (sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, new_feat_output_tuple, new_feature_mask_tuple, task_type_rpt), \
                inst_idx_to_position_map = collate_active_info((sequence_output_rpt, visual_output_rpt, input_ids_rpt, input_mask_rpt, video_mask_rpt, new_feat_output_tuple, new_feature_mask_tuple, task_type_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device
                                                               )


            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]

            # pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, args.max_words) # hard-code with 30 as max-length
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()
    
            "Save Intermediate Results, Do Squeeze on irregular shapes"
            new_batch_hpy = []
            for item in batch_hyp:
                new_item = item[0]
                new_item.extend([0] * (args.max_words - len(new_item)))
                new_batch_hpy.append(new_item)
            all_gt_lists.append(caption_list)
            all_rst_lists.append(new_batch_hpy)

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                
                if args.t1_postprocessing:
                    match = re.search(r'action[0-9]+[ ]*', decode_text)
                    if match !=None:

                        match_action_type = match.group(0).replace(' ','')
                        replace_type = action_token2full_description[match_action_type].replace(' unknown','').replace('made shot ','').replace('missed shot','miss')
                        decode_text  = decode_text.replace(match_action_type, replace_type)
                result_list_byTask[task_type.tolist()[re_idx]].append(decode_text)
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip("##").strip()
                caption_list_byTask[task_type.tolist()[re_idx]].append(decode_text)
                all_caption_lists.append(decode_text)
    
    """ Process the whole results """
    final_pred = np.stack(all_rst_lists, 0).reshape(-1, args.max_words)
    final_gt = np.concatenate(all_gt_lists, 0)

    sr_list = []
    acc_list = []
    mIoU = []
    mInter = []
    nonzero_pred_list = []
    nonzero_gt_list = []

    """ Calculate the metrics all together """
    for pred, gt in zip(final_pred, final_gt):
        if len(pred.shape) > 1: # make sure the shape is good;
            pred = np.squeeze(pred)

        nonzero_mask = np.where(gt > 0, 1, 0)
        assert np.sum(nonzero_mask) != 0, "All groundtruth is zeors!!!!!!!!!!!"
        nonzero_gt = gt[np.nonzero(gt)][:-1] # get rid of padding and 102 (End-of_Seq) token
        # nonzero_pred  = pred[np.nonzero(pred * nonzero_mask)][:-1] # get rid of padding and 102 (End-of_Seq) token
        nonzero_pred = pred[:len(nonzero_gt)]

        sr_list.append(success_rate(np.expand_dims(nonzero_pred, 0), np.expand_dims(nonzero_gt, 0)))
        nonzero_pred_list.extend(nonzero_pred.tolist())
        nonzero_gt_list.extend(nonzero_gt.tolist())
        # acc_list.append(mean_category_acc(nonzero_pred.tolist(), nonzero_gt.tolist()))
        mIoU.append(acc_iou_onehot(np.expand_dims(nonzero_pred, 0), np.expand_dims(nonzero_gt, 0)))
        score = []
        for item in nonzero_pred:
            if item in nonzero_gt:
                score.append(1.0)
            else:
                score.append(0.0)
        if len(score) > 0:
            mInter.append(sum(score) / len(score))
            
    acc_list.append(mean_category_acc(nonzero_pred_list, nonzero_gt_list))

    # Save full results
    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    # Save pure results
    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    # Evaluate
    for task in test_tasks:
        r  = [caption_list_byTask[task]]
        h  = result_list_byTask[task]
        # Is this the place where I should put the Precision/Recall/AUC curve?

        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=r, hyp_list=h)
        logger.info(">>> TASK {:d}: BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                    format(task, metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
        logger.info(">>> TASK {:d}: METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(task, metrics_nlg["METEOR"], metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"]))

        Bleu_4 = metrics_nlg["Bleu_4"]

    return sum(sr_list)/ len(sr_list), sum(acc_list)/ len(acc_list), sum(mIoU)/ len(mIoU), sum(mInter) / len(mInter)

DATALOADER_DICT = {}
DATALOADER_DICT["youcook"] = {"train":dataloader_youcook_train, "val":dataloader_youcook_test}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test}
DATALOADER_DICT["ourds"] = {"train":dataloader_ourds_train, "val":dataloader_ourds_test}

action_list = json.load(open('./data/action_list.json', 'r'))
action_token2full_description = {'action%s'%a_idx:a_l.lower().replace('_',' ').replace('-',' ') for a_idx, a_l in enumerate(action_list)}

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)
    n_gpu = 1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ## Collect declared feature, to initialize model + dataloader
    feature_tuple = {}
    # for arg in vars(args):
    #     # print(arg)
    #     if '_features_path' in arg:
    #         feature_tuple[arg.split('_')[0]] = getattr(args, arg)
    # logger.info("***** Using the following Features %s *****", feature_tuple)
    num_token = len(feature_tuple) + 1 # Timesformer feature + others
    ## Collect declared feature, to initialize model + dataloader
    """
    Mannually use the default fine-grained features, which are
    1. ball_basket_cl2_sum
    2. courtline segmentation
    3. Use courtseg as a symbol
    """
    feature_tuple['courtseg'] = './data/cls2_ball_basket_sum_concat_original_courtline_fea.pickle'

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer_original = BertTokenizer.from_pretrained(args.bert_model+'-original', do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank, type_vocab_size=num_token)
    
    for action_token, action_description in action_token2full_description.items():
        ids = tokenizer_original.convert_tokens_to_ids(tokenizer_original.tokenize(action_description))
        random_action_embed = model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])]
        new_action_embed = torch.mean(model.bert.embeddings.cpu()(torch.tensor([ids])),dim=1)
        #new_action_embed = new_action_embed.to(random_action_embed.device)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(random_action_embed.cpu(), new_action_embed)
        #print(output)
        with torch.no_grad():
            model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])] = new_action_embed
            model.decoder.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])] = new_action_embed
        random_action_embed = model.bert.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])]
        output = cos(random_action_embed, new_action_embed)
        # print(output)
        output = cos(model.decoder.embeddings.word_embeddings.weight[tokenizer.convert_tokens_to_ids([action_token])], new_action_embed)
        # print(output)
        #print(tokenizer_original.convert_ids_to_tokens(ids))
    model.to(device)
    model.bert.to(device)
    model.bert.embeddings.to(device)
    model.bert.embeddings.word_embeddings.to(device)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if 'bert' in name:
    #             print(name)
    #     else:
    #         print(name)
    assert args.task_type == "caption"
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=None)

    assert args.datatype in DATALOADER_DICT
    args.video_feature = pickle.load(open(args.features_path, 'rb'))
    args.feature_tuple = {}
    # breakpoint()
    common_keys = []
    for name, path in feature_tuple.items():
        args.feature_tuple[name] = []
        logger.info(" Loading feature pickle from %s", path)

        # if name == 'allplayer':
        #     # Use online-reading, thus there is None pickle to the list 
        #     args.feature_tuple[name].append(None)
        # else:
        args.feature_tuple[name].append(pickle.load(open(path, 'rb')))
        assert name in ['bbx', 'courtseg', 'keypoint', 'allplayer', 'bbxcls2', 'bbxball', 'bbxbasket']

        if name == 'bbx' or name == 'bbxcls2' or name == 'bbxball' or name == 'bbxbasket':
            args.feature_tuple[name][0]['video10084'] = np.zeros((args.max_frames, 768)) # Add this to avoid key-error
            args.feature_tuple[name].append((args.max_frames, 768)) # [numFrame, dimFeature]
        elif name == 'courtseg':
            args.feature_tuple[name].append((args.max_frames, 768 * 2))
        elif name == 'keypoint':
            args.feature_tuple[name][0]['video22693'] = np.zeros((args.max_frames, 768)) # Add this to avoid key-error
            args.feature_tuple[name][0]['video5273'] = np.zeros((args.max_wmax_framesords, 768)) # Add this to avoid key-error
            args.feature_tuple[name].append((args.max_frames, 768))
        else:
            args.feature_tuple[name].append((args.max_frames, 10, 768)) # For allplayer, [numFrame, numPlayer, dimFeature]

    "Check the common keys in all feature_tuple"
    # print("non-commone elements in features: {}".format([x for x in aa if x not in bbb]))
    # breakpoint()

    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, split_type='val')
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, split_type='test')

    if args.local_rank == 0:
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        debug_eval = False

        if debug_eval is True:
            sr, acc, mIoU, mInter = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
            # sr, acc, mIoU = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            if debug_eval is False:
                tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)
            else:
                tr_loss = 0
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, type_name="")
                if epoch > 0:
                    sr, acc, mIoU, mInter = eval_epoch(args, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    if best_score <= acc:
                        best_score = acc
                        best_output_model_file = output_model_file
                        logger.info('This is the best model in val set so far, testing test set....')
                        eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                    # logger.info("The best model is: {}, the Best-Acc {}, SR {}, mIoU {}, mInter {}".format(best_output_model_file, best_score, sr, mIoU, mInter))
                    logger.info("The best model is: {}, the Best-Acc {}, SR {}, mIoU {}".format(best_output_model_file, best_score, sr, mIoU))

                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')
            model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
    elif args.do_eval:
        if args.local_rank == 0:

            test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer,split_type='test')
            model = load_model(-1, args, n_gpu, device, model_file='/home/ubuntu/vcap/content/ckpts/ckpt_ourds_caption/pytorch_model.bin.4')
            eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    main()