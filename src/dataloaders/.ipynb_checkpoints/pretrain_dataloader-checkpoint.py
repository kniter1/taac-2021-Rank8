# %%
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
import time
from modules.tokenization import BertTokenizer

class TAGGING_Pretrain_DataSet(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            video_path,
            video_caption_path,
            video_features_path,
            audio_feature_path,
            tokenizer,
            max_words=400,
            feature_framerate=1.0,
            max_frames=200,
            max_sequence=200
    ):
        self.video_list = self._get_video_id_list(video_path)
        self.video_caption_dict = self._get_video_caption(video_caption_path)
        self.video_feature_dict = self._get_feature_dict(video_features_path)
        self.audio_feature_dict = self._get_feature_dict(audio_feature_path)
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.max_sequence = max_sequence
        self.tokenizer = tokenizer
        self.video_feature_size = 1024
        self.audio_feature_size = 128

    def __len__(self):
        return len(self.video_list)
    
    def _get_video_id_list(self, video_path):
        video_id_list = []
        video_list = os.listdir(video_path)
        for video in video_list:
            video_id = video.split('.mp4')[0]
            video_id_list.append(video_id)

        return video_id_list
    
    def _get_feature_dict(self, feature_path):
        feature_dict = {}
        features_list = os.listdir(feature_path)
        for feature in features_list:
            name = feature.split('.npy')[0]
            if 'b0f487ea8a4fc44003c7e05e3afee3c9' in feature and 'audio' in feature_path:
                feature_dict['b0f487ea8a4fc44003c7e05e3afee3c9'] = np.zeros((20,128), dtype=np.float)
                continue
            feature = np.load(os.path.join(feature_path, feature))
            feature_dict[name] = feature
        
        return feature_dict
    
    def _get_video_caption(self, video_caption_path):
        video_caption_dict = {}
        caption_list = os.listdir(video_caption_path)
        for video_caption in caption_list:
            files = open(os.path.join(video_caption_path, video_caption), 'r', encoding='utf-8')
            line = files.readline()
            caption = json.loads(line)
            caption = caption['video_asr']
            video_id = video_caption.split('.txt')[0]
            video_caption_dict[video_id] = caption

        return video_caption_dict   
    


    def _get_text(self, video_id, caption): #MSRVTT数据集一个video对应好几个句子，所以这里使用了加载

        words = self.tokenizer.tokenize(caption)
        words = ["[CLS]"] + words
        total_length_with_CLS = self.max_words - 1
        if len(words) > total_length_with_CLS:
       #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!INFO words > total_length {}'.format(len(words)))
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
        masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        input_mask = [1] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            masked_token_ids.append(0)
            token_labels.append(-1)
        assert len(input_ids) == self.max_words
        assert len(input_mask) == self.max_words
        assert len(masked_token_ids) == self.max_words
        assert len(token_labels) == self.max_words


        pairs_text = np.array(input_ids)
        pairs_mask = np.array(input_mask)
        pairs_masked_text = np.array(masked_token_ids)
        token_labels = np.array(token_labels)


        return pairs_text, pairs_mask, pairs_masked_text, token_labels

    def _get_video(self, video_id):
        video_mask = np.zeros((self.max_frames), dtype=np.long)

        video = np.zeros((self.max_frames, self.video_feature_size), dtype=np.float)

        video_slice = self.video_feature_dict[video_id]

        if self.max_frames < video_slice.shape[0]:
         #   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!INFO: video_slice > max_frames {}'.format(video_slice.shape[0]))
            video_slice = video_slice[:self.max_frames]

        slice_shape = video_slice.shape

        if len(video_slice) < 1:
            print("video_id: {}".format(video_id))
        else:
            video[:slice_shape[0]] = video_slice

        v_length = slice_shape[0]
        video_mask[:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = []
        masked_video = video.copy()
        for i, video_frame in enumerate(masked_video):
            if i < v_length:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    masked_video[i] = [0.] * video.shape[-1]
                    video_labels_index.append(i)
                else:
                    video_labels_index.append(-1)
            else:
                video_labels_index.append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.long)
        # -----> Mask Frame Model
        
        return video, video_mask, masked_video, video_labels_index

    def _get_audio(self, audio_id):
        audio_mask = np.zeros((self.max_sequence), dtype=np.long)
        audio = np.zeros((self.max_sequence, self.audio_feature_size), dtype=np.float)

        audio_slice = self.audio_feature_dict[audio_id]
        if self.max_sequence < audio_slice.shape[0]:
          #  print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!INFO: audio_sice> max_sequence{}'.format(audio_slice.shape[0]))
            audio_slice = audio_slice[:self.max_sequence]
        
        slice_shape = audio_slice.shape
        if len(audio_slice) < 1:
            print('audio_id: {}'.format(audio_id))
        else:
            audio[:slice_shape[0]] = audio_slice
        a_length = slice_shape[0]
        audio_mask[:a_length] = [1] * a_length
        # Mask Frame Model <-----
        audio_labels_index = []
        masked_audio = audio.copy()
        for i, audi_frame in enumerate(masked_audio):
            if i < a_length:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    masked_audio[i] = [0.] * audio.shape[-1]
                    audio_labels_index.append(i)
                else:
                    audio_labels_index.append(-1)
            else:
                audio_labels_index.append(-1)
        audio_labels_index = np.array(audio_labels_index, dtype=np.long)
        # -----> Mask Frame Model

        return audio, audio_mask, masked_audio, audio_labels_index





    def __getitem__(self, idx):
        video_id = self.video_list[idx]

        caption = self.video_caption_dict[video_id]
 
        text, text_mask, masked_text, text_token_labels = self._get_text(video_id, caption)
        video, video_mask, masked_video, video_token_labels = self._get_video(video_id)
        audio, audio_mask, masked_audio, audio_token_labels = self._get_audio(video_id)

        return text, text_mask, masked_text, text_token_labels, \
            video, video_mask, masked_video, video_token_labels, \
                audio, audio_mask, masked_audio, audio_token_labels