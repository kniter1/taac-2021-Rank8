from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
import time
import argparse
from src.models.tokenization import BertTokenizer
from src.models.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from src.models.modeling import Tagging_UniVL
from src.models.optimization import BertAdam
from torch.utils.data import DataLoader
import torch.utils.data as data
from util import parallel_apply, get_logger
from src.dataloaders.tagging_dataloader import TAGGING_DataSet
from src.dataloaders.tagging_dataloader import TAGGING_Train_DataSet
#torch.distributed.init_process_group(backend="nccl")

global logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader_tagging_test(args, tokenizer):
    tagging_testset = TAGGING_DataSet(
        video_path=args.video_path,
        label_path=args.label_path,
        video_caption_path=args.video_caption_path,
        video_features_path=args.video_features_path,
        audio_feature_path=args.audio_features_path,
        tokenizer=tokenizer,
        max_words=args.max_words,
        feature_framerate=1.0,
        max_frames=args.max_frames,
        max_sequence=args.max_sequence
    )

    dataloader_tagging = DataLoader(
        tagging_testset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    return  tagging_testset ,dataloader_tagging, len(tagging_testset)

def load_model(flod, args, n_gpu, device, model_file=None): #模型加载

    model_file = os.path.join(model_file, "pytorch_model_{}flod.bin.".format(flod))
    logger.info("**** loading model_file=%s *****", model_file)

    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        model = Tagging_UniVL.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model,cache_dir=args.cache_dir,
                                        state_dict=model_state_dict, task_config=args)


        model.to(device)
        logger.info('***** loading model successful! *****')
    else:
        model = None
    return model

#save_path = '/home/tione/notebook/test_5k_2nd_cross_embedding'
def model_test(args, model_list, test_dataset, test_dataloader, device, n_gpu):

    with torch.no_grad():
        output_dict = {}
        for index, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, video, video_mask, audio, audio_mask= batch
            pred_scores = torch.zeros((1, 82)).to(device)
            for model in model_list:
                model = model.to(device)
                model.eval()
                pred_scores += model.interfence(input_ids, input_mask, video, video_mask, audio, audio_mask)
            pred_scores = pred_scores / len(model_list)

          #  cross_embedding = cross_embedding.data.cpu().numpy()
            video_id = test_dataset.video_list[index]
          #  np.save(os.path.join(save_path, video_id+'.npy'), cross_embedding)
            output_dict[video_id+'.mp4'] = get_output(index, test_dataset, pred_scores)
            logger.info('{}/{} interfence finished'.format(index, len(test_dataloader)))
        
    write2json(args, output_dict)
           

def get_output(index, test_dataset, pred_scores, top_k=20):
    results = []
    results_labels_scores = {}
    video_id = test_dataset.video_list[index]
    labes_scores_dict = {}
    pred_scores=  list(pred_scores.view(-1))
    for ids, scores in enumerate((pred_scores)):
        labes_scores_dict[float(scores)] = ids
    

    pred_scores = sorted(pred_scores, reverse=True)
    pred_scores = pred_scores[:top_k]
    labels = []
    scores = []
    for score in pred_scores:
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
    output_json_path = args.output_json
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False)


parser = argparse.ArgumentParser(description="model interfence")
parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
parser.add_argument("--do_train", action='store_true', help="Whether to run training.") 
parser.add_argument("--do_test", action='store_true', help="whether to run test")
parser.add_argument("--model_file", type=str, help="model store path")
parser.add_argument("--output_json", type=str, help="output files")
#parser.add_argument("--epoch", type=int, help="choice nums models")
parser.add_argument('--video_path', type=str, default='D:\Master\比赛\\tagging\\testfile\\video', help='')
parser.add_argument('--label_path', type=str, default='D:\Master\比赛\\tagging\\testfile\label_id\label.txt', help='')
parser.add_argument('--video_features_path', type=str, default='D:\Master\比赛\\tagging\\testfile\\feature', help='feature path')
parser.add_argument('--audio_features_path', type=str, default='')
parser.add_argument('--video_caption_path', type=str, default='D:\Master\比赛\\tagging\\testfile\caption', help='')
parser.add_argument('--max_words', type=int, default=400, help='')
parser.add_argument('--max_frames', type=int, default=200, help='')
parser.add_argument('--max_sequence', type=int, default=200, help='')
parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
parser.add_argument('--audio_model', default="audio-base", type=str, required=False, help='AUdio module')
parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
parser.add_argument("--bert_model", default="bert-base-chinese", type=str, required=False,
                        help="Bert pre-trained model")
parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
parser.add_argument("--num_labels", type=int, default=82, required=False)
parser.add_argument('--video_dim', type=int, default=1024, required=False,help='video feature dimension')
parser.add_argument('--audio_dim', type=int, default=128, required=False, help='')
parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer NO. of audio")
parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
parser.add_argument('--cache_dir', type=str, help='bert cache dir')
args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained(args.bert_model)
n_gpu = 1
logger = get_logger(os.path.join("/home/tione/notebook/cmy/tione/notebook/univl/tagging_unvil/tagging/UniVL/output_json", "log.txt"))

if args.local_rank ==0:
    logger.info("***** dataloader loading *****")
    test_dataset, test_dataloader, test_length = dataloader_tagging_test(args, tokenizer)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch size = %d", 1)
    logger.info("  Num steps = %d", len(test_dataloader))
    model_list = []
    k_fold=5
    for current_fold in range(k_fold):       
        model = load_model(current_fold, args, n_gpu, device, model_file=args.model_file)
        model_list.append(model)
        print('**************************** model_{}\{}fold loading'.format(current_fold+1, k_fold))
    model_test(args, model_list, test_dataset, test_dataloader, device, n_gpu)
    logger.info("  Interfence Finshed")


