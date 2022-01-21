from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
from torch.utils.data import (SequentialSampler)
import numpy as np
import random
import os
from metrics import compute_metrics
import time
import argparse
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import Tagging_UniVL
from modules.optimization import BertAdam
from torch.utils.data import DataLoader
import torch.utils.data as data
from util import parallel_apply, get_logger
from dataloaders.pretrain_dataloader import TAGGING_Pretrain_DataSet
#torch.distributed.init_process_group(backend="nccl")
import time
global logger

def get_args(description='tagging pretrain task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", action='store_true', help="whether to run test")

    parser.add_argument('--video_path', type=str, default='D:\Master\比赛\\tagging\\testfile\\video', help='')
    parser.add_argument('--video_features_path', type=str, default='D:\Master\比赛\\tagging\\testfile\\feature', help='feature path')
    parser.add_argument('--video_caption_path', type=str, default='D:\Master\比赛\\tagging\\testfile\caption', help='')
    parser.add_argument('--audio_features_path', type=str, default='', help='')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='') #读线程
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')#初始化学习率
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit') #总共epoch数
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')#训练集bh
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') # 学习率衰减
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--audio_dim', type=int, default=128, help='audio_feature_dimension') 
    parser.add_argument('--seed', type=int, default=66, help='random seed') 
    parser.add_argument('--max_words', type=int, default=400, help='')
    parser.add_argument('--max_frames', type=int, default=400, help='')
    parser.add_argument('--max_sequence', type=int, default=200, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--output_json_file", default=None, type=str, required=False,
                        help="The test output json file where the model prediction")
    parser.add_argument("--bert_model", default="bert-base-chinese", type=str, required=True,
                        help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="audio-base", type=str, required=False, help="Audio module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") #是否加载模型
    parser.add_argument("--bert_model_path", default=None, type=str, required=False, help="bert model inital")
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

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")# 分布式训练
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") #分布式训练
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument("--datatype", default="tagging", type=str, help="Point the dataset `tagging` to finetune.")
    parser.add_argument("--num_labels", default=82, type=int, help="tagging labels")

    args = parser.parse_args()

    if args.sampled_use_mil:  # sample from each video, has a higher priority than use_mil.
        args.use_mil = True

    # Check paramenters
    if not args.do_pretrain:
        raise ValueError("`do_pretrain` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  #  world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
  #  args.world_size = world_size
 #   print('---------------INFO--------------- world_size:{}'.format(world_size))

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0: #batch_size必须整除gpu个数
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank): #初始化模型 加载预训练UniVL

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu') #如果需要加载模型
        for k in list(model_state_dict.keys()):
            k_size = k.split('.')
            if k_size[0] == 'bert':
                del model_state_dict[k]
    else:
        model_state_dict = None
    
    if args.bert_model_path:
        bert_state_dict = torch.load(args.bert_model_path, map_location='cpu')

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = Tagging_UniVL.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, bert_state_dict=bert_state_dict, task_config=args)
    

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

  #  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                #      output_device=local_rank, find_unused_parameters=True)

    
    return optimizer, scheduler, model


def dataloader_pretrain(args, tokenizer,):

    logger.info('***** loading data *****')
    tagging_dataset = TAGGING_Pretrain_DataSet(
        video_path=args.video_path,
        video_caption_path=args.video_caption_path,
        video_features_path=args.video_features_path,
        audio_feature_path=args.audio_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_sequence=args.max_sequence
    )
  #  train_sampler = torch.utils.data.distributed.DistributedSampler(tagging_dataset)

    train_dataloader = DataLoader(
        tagging_dataset,
        num_workers=args.num_thread_reader,
        batch_size=args.batch_size // args.n_gpu,
        pin_memory=False,
        shuffle=True,
       # sampler = train_sampler,
        drop_last=True,
    )

    return train_dataloader, len(tagging_dataset)


def save_model(epoch, args, model, type_name=""): #模型存储
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None): #模型加载
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

        model = Tagging_UniVL.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache() #释放缓存分配器当前持有的且未占用的缓存显存,供gpu其他程序使用
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0


    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        text, text_mask, masked_text, text_token_labels, \
            video, video_mask, masked_video, video_token_labels, \
                audio, audio_mask, masked_audio, audio_token_labels = batch
        
        pretrain_loss = model(text, text_mask, video, video_mask, audio, audio_mask, \
            masked_text, text_token_labels, masked_video, video_token_labels, \
            masked_audio, audio_token_labels, training=True)

        if n_gpu > 1:
            model_loss = pretrain_loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            model_loss = pretrain_loss / args.gradient_accumulation_steps

        pretrain_loss.backward()

        total_loss += float(pretrain_loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #防止梯度爆炸

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(pretrain_loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    model = init_model(args, device, n_gpu, args.local_rank)   #args把参数传进去
    model = model.to(device)
    print('loading successful!')

    train_dataloader, train_length = dataloader_pretrain(args, tokenizer)
    print('***** dataloader loading successful *****')
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs
    
    global_step = 0

    coef_lr = args.coef_lr
    if args.init_model:
        coef_lr = 1.0
    
    print('prep optimizer...')
    optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
    print('optimizer finished')

    if args.local_rank == 0:
        logger.info("***** Running pretraining *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)


    for epoch in range(args.epochs):
        print('start traing...')
        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                           scheduler, global_step, local_rank=args.local_rank)

        if args.local_rank == 0:
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            save_model(epoch, args, model, type_name="pretrain")

if __name__ == "__main__":
    main()