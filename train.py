from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
import sys
import time
import argparse
from src.models.tokenization import BertTokenizer
from src.models.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from src.models.modeling import Tagging_UniVL
from src.models.optimization import BertAdam
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as data
from util import parallel_apply, get_logger
from src.dataloaders.tagging_dataloader import TAGGING_DataSet
from src.dataloaders.tagging_dataloader import TAGGING_Train_DataSet
#torch.distributed.init_process_group(backend="nccl")

global logger
"""
group
进程组。默认情况只有一个组，一个 job 为一个组，也为一个 world

world size
全局进程个数

rank
表示进程序号，用于进程间的通讯。rank=0 的主机为 master 节点

local rank
进程内 GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。 rank=3, local_rank=0 表示第 3 个进程内的第 1 块 GPU。
"""


def get_args(description='tagging'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", action='store_true', help="whether to run test")

    parser.add_argument('--video_path', type=str, default='D:\Master\比赛\\tagging\\testfile\\video', help='')
    parser.add_argument('--label_path', type=str, default='D:\Master\比赛\\tagging\\testfile\label_id\label.txt', help='')
    parser.add_argument('--label_info_path', type=str, default='D:\Master\比赛\\tagging\\testfile\label_info\labe_info.txt', help='')
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
    parser.add_argument('--seed', type=int, default=666, help='random seed') 
    parser.add_argument('--max_words', type=int, default=400, help='')
    parser.add_argument('--max_frames', type=int, default=200, help='')
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
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")# 分布式训练
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") #分布式训练
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")
    parser.add_argument("--do_lower_case",action='store_true',default=False, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--datatype", default="tagging", type=str, help="Point the dataset `tagging` to finetune.")
    parser.add_argument("--num_labels", default=82, type=int, help="tagging labels")
    parser.add_argument("--k_fold", default=5, type=int, help="tagging k_fold")

    args = parser.parse_args()
    # Check paramenters
    if args.gradient_accumulation_steps < 1: #判断梯度累计步骤，一般设为1
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args): #设置随机种子，防止每次初始化参数不同
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  #这个设置为true，就可以保证每次算法返回值是固定得

  #  world_size = torch.distributed.get_world_size() #world_size, 全局进程个数
 #   print('---------------INFO--------------- world_size:{}'.format(world_size))
    torch.cuda.set_device(args.local_rank) #进程内 GPU 编号，非显式参数，由 torch.distributed.launch 内部指定。 rank=3, local_rank=0 表示第 3 个进程内的第 1 块 GPU。
  #  args.world_size = world_size

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

    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0: #batch_size必须整除gpu个数
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank): #初始化模型 加载预训练UniVL

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu') #如果需要加载模型

    else:
        model_state_dict = None
    bert_state_dict=None
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
    
    freeze_layer = ['bert.encoder.layer.0.', 'bert.encoder.layer.1.', 'bert.encoder.layer.2.', 'bert.encoder.layer.3.','bert.encoder.layer.4.','bert.encoder.layer.5.']
    for n, p in model.named_parameters():
        for freeze in freeze_layer:
            if freeze in n:
                p.requires_grad = False

    param_optimizer = list(model.named_parameters())
    param_optimizer = list(filter(lambda p:p[1].requires_grad, param_optimizer))

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    
  #  print('***** local rank *****', local_rank)
 #   model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                     # output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def get_k_fold_data(k_fold, current_fold, video_list):  

    fold_size = len(video_list) // k_fold  # 每份的个数:数据总条数/折数（组数）
    train_video= None
    for j in range(k_fold):
        idx = slice(j * fold_size, (j + 1) * fold_size)#slice(start,end,step)切片函数
        ##idx 为每组 valid
        part = video_list[idx]
        if j == current_fold: ###第i折作valid
            val_video = part
        elif train_video is None:
            train_video = part
        else:
            train_video = train_video + part #dim=0增加行数，竖着连接


    return train_video, val_video

def get_video_id_list(args):
    video_path = args.video_path
    video_id_list = []
    video_list = os.listdir(video_path)
    for video in video_list:
        video_id = video.split('.mp4')[0]
        video_id_list.append(video_id)

    return video_id_list

def dataloader_tagging_train(args, tokenizer, video_list, current_fold):
    train_video_list, val_video_list = get_k_fold_data(args.k_fold, current_fold, video_list)  
    train_dataset = TAGGING_Train_DataSet(
        video_list=train_video_list,
        label_path=args.label_path,
        video_caption_path=args.video_caption_path,
        video_features_path=args.video_features_path,
        audio_feature_path=args.audio_features_path,
        label_info_path=args.label_info_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_sequence=args.max_sequence,
    )
    val_dataset = TAGGING_Train_DataSet(
        video_list=val_video_list,
        label_path=args.label_path,
        video_caption_path=args.video_caption_path,
        video_features_path=args.video_features_path,
        audio_feature_path=args.audio_features_path,
        label_info_path=args.label_info_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_sequence=args.max_sequence,
    )
 #   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
 #   val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
    #    sampler=sampler,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
 #       sampler=val_sampler,
        drop_last=True,
        
    )
    train_length = len(train_dataset)
    val_length = len(val_dataset)

    return train_dataloader, val_dataloader, train_length, val_length

def dataloader_tagging_test(args, tokenizer):
    tagging_testset = TAGGING_DataSet(
        video_path=args.video_path,
        label_path=args.label_path,
        video_caption_path=args.video_caption_path,
        video_features_path=args.video_feaures_path,
        audio_features_path=args.audio_feature_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_sequence=args.max_sequence
    )

    dataloader_tagging = DataLoader(
        tagging_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )

    return  tagging_testset ,dataloader_tagging, len(tagging_testset)



def save_model(args, model, current_fold, type_name=""): #模型存储
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model_{}flod.bin.{}".format(current_fold, "" if type_name=="" else type_name+"."))
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

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0): #训练流程
    global logger
    torch.cuda.empty_cache() #释放缓存分配器当前持有的且未占用的缓存显存,供gpu其他程序使用
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_gap = 0
    
    for step, batch in enumerate(train_dataloader):
        #正常训练
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label = batch
        text_loss, visual_loss, audio_loss, cross_loss, model_loss, \
        text_gap, visual_gap, audio_gap, cross_gap, gap_score = model(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, ground_trunth_labels=ground_label, training=True)
        if n_gpu > 1:
            model_loss = model_loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            model_loss = model_loss / args.gradient_accumulation_steps
        model_loss.backward() #反向传播，得到正常的grad
        total_loss += float(model_loss)
        total_gap += gap_score

        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #防止梯度爆炸

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%d, Step: %d/%d, Lr: %s, Text_Loss: %f, Visual_Loss: %f, Audio_Loss: %f, Cross_Loss: %f, Totual_Loss: %f,  Time/step: %f, text_gap: %f, viusal_gap: %f, audio_gap: %f, cross_gap: %f, gap: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(text_loss), float(visual_loss), float(audio_loss), float(cross_loss), float(model_loss), 
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps), text_gap, visual_gap, audio_gap, cross_gap, gap_score)
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    total_gap = total_gap / len(train_dataloader)
    return total_loss, global_step, total_gap


def eval_epoch(args, model, val_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        total_gap1 = 0.
        total_text_gap = 0.
        total_visual_gap = 0.
        total_audio_gap = 0.
        total_cross_gap = 0.
        for _, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            """
            模型inference阶段输出需要修改
            """

            input_ids, input_mask, video, video_mask, audio, audio_mask, ground_trunth_labels = batch
            text_gap, visual_gap, audio_gap, cross_gap, gap1 = model(input_ids, input_mask, video, video_mask, audio, audio_mask, ground_trunth_labels=ground_trunth_labels, training=False)
            total_gap1 += gap1
            total_text_gap += text_gap
            total_visual_gap += visual_gap
            total_audio_gap += audio_gap
            total_cross_gap += cross_gap

        total_gap1 = total_gap1 / len(val_dataloader)
        total_text_gap = total_text_gap / len(val_dataloader)
        total_visual_gap = total_visual_gap / len(val_dataloader)
        total_audio_gap = total_audio_gap / len(val_dataloader)
        total_cross_gap = total_cross_gap / len(val_dataloader)


        return total_text_gap, total_visual_gap, total_audio_gap, total_cross_gap, total_gap1

    



def model_test(args, model, test_dataset, test_dataloader, device, n_gpu):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output_dict = {}
        for index, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, video, video_mask, audio, audio_mask= batch
            pred_scores = model.interfence(input_ids, input_mask, video, video_mask, audio, audio_mask)
            video_id = test_dataset.video_list[index]
            output_dict[video_id+'.mp4'] = get_output(index, test_dataset, pred_scores)
        
    write2json(args, output_dict)
           

def get_output(index, test_dataset, pred_scores, top_k=20):
    results = []
    results_labels_scores = {}
    video_id = test_dataset.video_list[index]
    labes_scores_dict = {}
    for ids, scores in enumerate(pred_scores):
        labes_scores_dict[float(scores)] = ids
    
    pred_scores = sorted(pred_scores)
    pred_scores = scores[:top_k]
    labels = []
    scores = []
    
    for pred_score in pred_scores:
        score = float(score)
        label_ids = labes_scores_dict[score]
        label = test_dataset.index2label[label_ids]
        labels.append(label)
        scores.append(score)
    
    results_labels_scores["labels"] = labels
    results_labels_scores["scores"] = scores
    results.append(results_labels_scores)

    return  {"result": results}


def write2json(args, output_dict):
    output_json_path = args.output_json_file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f)

        
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    if args.local_rank == 0:
        logger.info("***** k_fold traing:{} *****".format(args.k_fold))
    
    video_list = get_video_id_list(args)
    
    for current_fold in range(args.k_fold):   
        if args.local_rank == 0:
            logger.info('***** {} fold strat *****'.format(current_fold + 1))
        model = init_model(args, device, n_gpu, args.local_rank)   #args把参数传进去
        model = model.to(device)
        print('loading successful!')
        if args.do_train:
            train_dataloader, val_dataloader, train_length, val_length = dataloader_tagging_train(args, tokenizer, video_list, current_fold)
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
            early_stop = 0
            for epoch in range(args.epochs):
                tr_loss, global_step, tr_gap = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                   scheduler, global_step,local_rank=args.local_rank)
                if args.local_rank == 0:
                    logger.info("Fold %d Epoch %d/%d Finished, Train Loss: %f, Train_gap: %f", current_fold+1, epoch + 1, args.epochs, tr_loss, tr_gap)
                if args.local_rank == 0:
                    logger.info("***** Running valing *****")
                    logger.info("  Num examples = %d", val_length)
                    logger.info("  Batch_size = %d", args.batch_size)
                    text_gap, visual_gap, audio_gap, cross_gap, gap1 = eval_epoch(args, model, val_dataloader, device, n_gpu)   
                    logger.info("----- val_dataset text_gap: %f, visual_gap: %f, audio_gap: %f, cross_gap: %f, gap: %f",  text_gap, visual_gap, audio_gap, cross_gap, gap1)
                    if best_score <=  gap1:
                        best_score = gap1
                        output_model_file = save_model(args, model, current_fold, type_name="")
                        best_output_model_file = output_model_file
                        early_stop=0
                    else:
                        early_stop += 1
                        if early_stop > 3:
                            break
                    logger.info("The best model is: {}, the gap is: {:.4f}".format(best_output_model_file, best_score))
                
                        
            if args.local_rank == 0:
                logger.info("{}/{} fold finished".format(current_fold+1, args.k_fold))
        

        
if __name__ == "__main__":
    main()

