# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import time

from .until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss
from .loss import Focalloss
from .module_bert import BertModel, BertConfig, BertOnlyMLMHead
from .module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from .module_audio import AudioModel, AudioConfig, AudioOnlyMLMHead
from .module_cross import CrossModel, CrossConfig
from .module_decoder import DecoderModel, DecoderConfig
from src.utils.tagging_eval import *

logger = logging.getLogger(__name__)



class Tagging_Classifier(nn.Module):
    def __init__(self, input_dims, nums_class=82, dropout=0.1):
        super(Tagging_Classifier, self).__init__()
        self.dense = nn.Linear(input_dims, nums_class)
        self.actvation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, cross_output):

        output = self.dense(cross_output)
        output = self.dropout(output)
        output = self.actvation(output)

        return output

    
class UniVLPreTrainedModel(PreTrainedModel, nn.Module):   #预训练模型
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, audio_config, cross_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)

        self.bert_config = bert_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.cross_config = cross_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.audio = None

    
    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name,  audio_model_name, cross_model_name, 
                        state_dict=None, bert_state_dict = None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):


        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

 
        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, audio_config, cross_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None
        assert model.audio is not None
        assert model.cross is not None
        assert model.va_cross is not None
        if task_config.do_pretrain:
            if bert_state_dict is not None:
                state_dict.update(bert_state_dict)

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
            
        return model

class NormalizeVideo(nn.Module):   #对视频特征进行layernorm 输入[B,L,D]
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video  #输出：[B, L, D]

class NormalizeAudio(nn.Module):   #对视频特征进行layernorm 输入[B,L,D]
    def __init__(self, task_config):
        super(NormalizeAudio, self).__init__()
        self.audio_norm2d = LayerNorm(task_config.audio_dim)

    def forward(self, audio):
        audio = torch.as_tensor(audio).float()
        audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
        audio = self.audio_norm2d(audio)
        return audio  #输出：[B, L, D]

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class Tagging_UniVL(UniVLPreTrainedModel): #UniVL模型
    def __init__(self, bert_config, visual_config, audio_config, cross_config, task_config):
        super(Tagging_UniVL, self).__init__(bert_config, visual_config, audio_config, cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings  #最大位置编码
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_sequence <= audio_config.max_position_embeddings
        #最大句子长度和最大帧数都要<跨模态位置编码
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings


        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder
        # Audio Encoder ====>
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                    self.task_config, "audio_num_hidden_layers")
        self.audio = AudioModel(audio_config)
        audio_word_embedding_weight = self.audio.embeddings.word_embeddings.weight
        # <====End of Audio_Encoder

        # Cross Encoder ===>
        cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
        self.cross = CrossModel(cross_config)
        self.va_cross = CrossModel(cross_config)
     #  self.at_cross = CrossModel(cross_config)
         # <=== End of Cross Encoder
        
        if self.task_config.do_pretrain:
            self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
            self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
            self.cls_audio = AudioOnlyMLMHead(audio_config, audio_word_embedding_weight)
            self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
            mILNCELoss = MILNCELoss(batch_size=task_config.batch_size, n_pair=1, )
            self._pretrain_sim_loss_fct = mILNCELoss
            self.loss_fct = CrossEn()
        

        self.cross_classifier = Tagging_Classifier(cross_config.hidden_size, task_config.num_labels) 
        self.text_classifier = Tagging_Classifier(bert_config.hidden_size, task_config.num_labels)
        self.visual_classifier = Tagging_Classifier(visual_config.hidden_size, task_config.num_labels)
        self.audio_classifier = Tagging_Classifier(audio_config.hidden_size, task_config.num_labels)    
        self.normalize_video = NormalizeVideo(task_config)
        self.normalize_audio = NormalizeAudio(task_config)
        
        self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask, video, video_mask, audio, audio_mask, 
                masked_text=None, text_token_labels=None, masked_video=None, video_token_labels=None,
             masked_audio=None, audio_token_labels=None, ground_trunth_labels=None,training=True, label_flag=None, alpha=0.0):

        input_ids = input_ids.view(-1, input_ids.shape[-1]) #文本输入维度[B, L] L=max_word_size
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) #[B, L]
        video_mask = video_mask.view(-1, video_mask.shape[-1])#[B,L] l=max_frames
        video = self.normalize_video(video)   #对video_dim进行layer_norm 输入[B, L, D]
        audio = self.normalize_audio(audio)

        text_output, visual_output, audio_output, \
            bert_pooled_output, visual_pooled_output, audio_pooled_output = self.get_text_visual_audio_output(input_ids,attention_mask, video, video_mask, audio, audio_mask) 

        
        
        
        """
        do pretrain:
        """ 
        if self.task_config.do_pretrain:
            loss = 0.

            masked_video = self.normalize_video(masked_video)
            masked_audio = self.normalize_audio(masked_audio)
    
            text_output_alm, visual_output_alm, audio_output_alm, _, _, _ = \
                self.get_text_visual_audio_output(masked_text,attention_mask, 
                                                    masked_video, video_mask, masked_audio, audio_mask)

            pooled_output, vat_cross_output, va_cross_output = self._get_cross_output(text_output_alm, visual_output_alm, audio_output_alm,
                                                                                        attention_mask, video_mask, audio_mask)
            text_cross_output, audio_cross_output, visual_cross_output = torch.split(vat_cross_output, [attention_mask.size(-1),audio_mask.size(-1), video_mask.size(-1)], dim=1)
            audio_cross_output_a,a_visual_cross_output = torch.split(va_cross_output, [audio_mask.size(-1), video_mask.size(-1)], dim=1)
            
            alm_loss = self._calculate_mlm_loss(text_cross_output, text_token_labels)
            loss += alm_loss
            
            nce_loss_v = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_token_labels)
            loss += nce_loss_v

            nce_loss_va = self._calculate_mfm_loss(a_visual_cross_output, video, video_mask, video_token_labels)
            loss += nce_loss_va

            nce_loss_a = self._calculate_mfm_loss(audio_cross_output, audio, audio_mask, audio_token_labels, video_flag=False)
            loss += nce_loss_a
            
            nce_loss_av = self._calculate_mfm_loss(audio_cross_output_a, audio, audio_mask, audio_token_labels, video_flag=False)
            loss += nce_loss_av
            
            sim_matrix_vt = self.get_similarity_logits(text_output, visual_output, attention_mask, video_mask,
                                                    shaped=True, _pretrain_joint=True)
            sim_loss_joint_vt = self._pretrain_sim_loss_fct(sim_matrix_vt)
            loss += sim_loss_joint_vt

            sim_matrix_va = self.get_similarity_logits(audio_output, visual_output, audio_mask, video_mask, shaped=True,
                                                        _pretrain_joint=True)
            sim_loss_joint_va = self._pretrain_sim_loss_fct(sim_matrix_va)
            loss += sim_loss_joint_va
            
            sim_matrix_text_visual = self.get_similarity_logits(text_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
            sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
            loss += sim_loss_text_visual
            
            sim_matrix_audio_visual = self.get_similarity_logits(audio_output_alm, visual_output_alm,
                                                                            audio_mask, video_mask, shaped=True)
            sim_loss_audio_visual = self.loss_fct(sim_matrix_audio_visual)
            loss += sim_loss_audio_visual

            return loss
        
        
        pooled_output, _, _ = self._get_cross_output(text_output, visual_output, audio_output, attention_mask, video_mask, audio_mask)
        cross_predict_scores = self.cross_classifier(pooled_output)
        text_predict_scores = self.text_classifier(bert_pooled_output)
        visual_predict_scores = self.visual_classifier(visual_pooled_output)
        audio_predict_scores = self.audio_classifier(audio_pooled_output)
        
        predict_scores = 0.5*((text_predict_scores+visual_predict_scores+audio_predict_scores)/3) + 0.5*cross_predict_scores
        

        
                                                            
        if training:


            loss = 0.
            text_loss = Focalloss(text_predict_scores, ground_trunth_labels)
            visual_loss = Focalloss(visual_predict_scores, ground_trunth_labels)
            audio_loss = Focalloss(audio_predict_scores, ground_trunth_labels)
            cross_loss = Focalloss(cross_predict_scores, ground_trunth_labels)
            text_predict_scores = text_predict_scores.data.cpu().numpy()
            visual_predict_scores = visual_predict_scores.data.cpu().numpy()
            audio_predict_scores = audio_predict_scores.data.cpu().numpy()
            cross_predict_scores = cross_predict_scores .data.cpu().numpy()
            predict_scores = predict_scores.data.cpu().numpy()
            ground_trunth_labels = ground_trunth_labels.data.cpu().numpy()
            text_gap = calculate_gap(text_predict_scores, ground_trunth_labels)
            visual_gap = calculate_gap(visual_predict_scores, ground_trunth_labels)
            audio_gap = calculate_gap(audio_predict_scores, ground_trunth_labels)
            cross_gap = calculate_gap(cross_predict_scores, ground_trunth_labels)
            gap_score = calculate_gap(predict_scores, ground_trunth_labels)
            loss = 0.1*text_loss + 0.1*visual_loss + 0.1*audio_loss + 0.7*cross_loss
            return text_loss, visual_loss, audio_loss, cross_loss, loss , \
                   text_gap, visual_gap, audio_gap, cross_gap, gap_score

        else:
            text_predict_scores = text_predict_scores.data.cpu().numpy()
            visual_predict_scores = visual_predict_scores.data.cpu().numpy()
            audio_predict_scores = audio_predict_scores.data.cpu().numpy()
            cross_predict_scores = cross_predict_scores .data.cpu().numpy()

            predict_scores = predict_scores.data.cpu().numpy()
            ground_trunth_labels = ground_trunth_labels.data.cpu().numpy()
            text_gap = calculate_gap(text_predict_scores, ground_trunth_labels)
            visual_gap = calculate_gap(visual_predict_scores, ground_trunth_labels)
            audio_gap = calculate_gap(audio_predict_scores, ground_trunth_labels)
            cross_gap = calculate_gap(cross_predict_scores, ground_trunth_labels)

            gap_score = calculate_gap(predict_scores, ground_trunth_labels)
            return text_gap, visual_gap, audio_gap, cross_gap, gap_score


    def get_text_visual_audio_output(self, input_ids, attention_mask, video, video_mask, audio, audio_mask):
        """
        输入维度：
        input_ids = [B, L]
        attention_mask = [B, L]
        video = [B, L, D]
        video_mask = [B, L]
        audio = [B, L, D]
        audio_mask = [B, L]
        """ #从联和输入中获取文字和视频输出
        

        encoded_layers, bert_pooled_output = self.bert(input_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1] #[B, L, 768], [B,768]

        visual_layers, visual_pooled_output = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]  #[B, L , 768], [B, 768]

        audio_layers, audio_pooled_output = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1] #[B, L, 768], [B, 768]

        return sequence_output, visual_output, audio_output, bert_pooled_output, visual_pooled_output, audio_pooled_output


    def _get_cross_output(self, sequence_output, visual_output,  audio_output, attention_mask, video_mask, audio_mask):
        
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        va_concat_mask = torch.cat((audio_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        audio_type_ =  torch.zeros_like(audio_mask)
  
        va_concat_type = torch.cat((audio_type_, video_type_), dim=1)
        va_cross_layers, va_pooled_output = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)

        va_cross_output = va_cross_layers[-1]
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        va_type_ = torch.ones_like(va_concat_mask)
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)
        vat_cross_layers, vat_pooled_output = self.cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
  
        return  vat_pooled_output, vat_cross_output, va_cross_output

    def interfence(self, input_ids, attention_mask, video, video_mask, audio, audio_mask):
        input_ids = input_ids.view(-1, input_ids.shape[-1]) #文本输入维度[B, L] L=max_word_size
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) #[B, L]
        video_mask = video_mask.view(-1, video_mask.shape[-1])#[B,L] l=max_frames
        video = self.normalize_video(video)   #对video_dim进行layer_norm 输入[B, L, D]
        audio = self.normalize_audio(audio)

        sequence_output, visual_output, audio_output, \
            bert_pooled_output, visual_pooled_output, audio_pooled_output = \
        self.get_text_visual_audio_output(input_ids,attention_mask, video, video_mask, audio, audio_mask) 
        pooled_output, _, _ = self._get_cross_output(sequence_output, visual_output,audio_output,  attention_mask, video_mask, audio_mask)
        cross_predict_scores = self.cross_classifier(pooled_output)
        text_predict_scores = self.text_classifier(bert_pooled_output)
        visual_predict_scores = self.visual_classifier(visual_pooled_output)
        audio_predict_scores = self.audio_classifier(audio_pooled_output)
        
        predict_scores = 0.5*((text_predict_scores+visual_predict_scores+audio_predict_scores)/3) + 0.5*cross_predict_scores

        return predict_scores
    
    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, output_alm, video, video_mask, video_labels_index, video_flag=True):
        if video_flag == True:
            afm_scores = self.cls_visual(output_alm)
        else:
            afm_scores = self.cls_audio(output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1]) #[B*L, D]

        video_tr = video.permute(2, 0, 1)  #[D, B, L]
        video_tr = video_tr.view(video_tr.shape[0], -1)# [D, B*L]
        logits_matrix = torch.mm(afm_scores_tr, video_tr) #[B*L, B*L]
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1)) #[B*L, B*L]
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8 #[b*l, b*l]

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss
    
    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,): #[B, L, 768], [b, l], [b,L]
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1) 
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)    
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out


    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
        retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits

