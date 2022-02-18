# TAAC-2021-神奈川冲浪里
## 目录结构
```
├── ckpts
│   └── ckpt
├── config
├── inference.py
├── inference.sh
├── __init__.py
├── pre
│   ├── extract_features.py
│   ├── extract_video_frame.py
│   ├── vision-transformer-pytorch
│   └── VIT_L_train_5k_features
├── pretrained_models
│   ├── bert_base_chinese
│   ├── UniVL_Pretrained_models
│   └── VIT-pretrained_models
├── pretrain.py
├── pretrain.sh
├── __pycache__
│   └── util.cpython-37.pyc
├── requirements.txt
├── src
│   ├── dataloaders
│   ├── modules
│   ├── __pycache__
│   └── utils
├── train.py
├── train.sh
└── util.py
```
## 环境配置
运行
```
bash init.sh
```
由于视频特征采用抽帧方式用VIT模型重新提取，因此需要首先克隆VIT的仓库，然后运行pre目录下面的两个脚本文件分别进行视频抽帧和提取特征：
```
bash extract_frame.sh
bash extract_features.sh
```

## 训练流程
### STEP1
根据网盘链接分别下载本次训练所需要的预训练模型  
VIT [链接](https://pretrained-models-1305291113.cos.ap-nanjing.myqcloud.com/imagenet21k%2Bimagenet2012_ViT-L_16.pth)  
pretrain_model[链接](https://pretrained-models-1305291113.cos.ap-nanjing.myqcloud.com/pytorch_model.bin.pretrain)  
接着创立两个文件夹，用于存放预训练模型：
```
mkdir taac-2021-S/pretrained_models/UniVL_Pretrained_models
mkdir mkdir taac-2021-S/pretrained_models/VIT_pretrained_models
```
将预训练模型移动到对应文件夹中  
我们采用的预训练模型在比赛数据集基础上进行了二次预训练，因此如果你需要在比赛数据集上自己预训练一遍，运行如下命令：
```
bash pretrain.sh
```  
### STEP2
首先，运行以下脚本文件
```
bash train.sh
```
该脚本文件路径需要修改的地方为:ROOT为根目录，默认为:/home/tione/notebook  
1.VIDEO_PATH：视频所在目录，默认为:algo-2021/dataset/videos/video_5k/train_5k  
2.LABEL_PATH：标签id所在文件，默认为:algo-2021/dataset/label_id.txt  
3.LABEL_INFO_PATH：训练集的真实标签，默认为：algo-2021/dataset/tagging/GroundTruth/tagging_info.txt  
4.VIDEO_FEATURES_PATH：训练集视频特征所在目录，默认为：taac-2021-神奈川冲浪里/pre/5.VIT_L_train_5k_features  
5.AUDIO_FEATURES_PATH：训练集音频特征所在目录，默认为：algo-2021/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging  
6.VIDEO_CAPTION_PATH：训练集视频文本描述：默认为：algo-2021/dataset/taggingtagging_dataset_train_5k/text_txt/tagging  
INIT_MODEL：训练所加载的预训练模型，默认为：taac-2021-神奈川冲浪里/pretrained_models/UniVL_Pretrained_models/pytorch_model.bin.pretrain  
预计训练时间为：3h
## 测试流程
### STEP1
首先进入pre目录，分别创建两个文件夹
```
mkdir taac-2021-S/pre/test_2nd_5k_384_frame_npy
mkdir taac-2021-S/pre/VTI_L_test_5k_2nd_features
```
运行
```
bash extract_frame.sh
```
其中VIDEO_PATH为测试集5k视频路径，默认为：algo-2021/dataset/videos/test_5k_2nd  
用来抽取视频帧数，预计抽取时间为：4h  
### STEP2
接着继续在该目录下运行
```
bash extract_features.sh
```
用来对视频特征进行抽取，
预计特征抽取时间为：4h  
### STEP3
最后在主目录下面运行
```
inference.sh
```
其中该脚本中的MODEL_FILE为训练后的模型所在目录，默认为：taac-2021-神奈川冲浪里/ckpts/ckpt
VIDEO_FEATURES_PATH为刚刚对视频特征提取后保存的目录，默认为：taac-2021-神奈川冲浪里/pre/VTI_L_test_5k_2nd_features    
测试时间：预计半小时
最终输入文件位于主目录下：test.json中  
## 补充说明
比赛所使用的预训练模型分别为VIT, UniVL, bert-baee-chinese  
预训练模型所在github链接为：
VIT [github](https://github.com/asyml/vision-transformer-pytorch)  
UniVL [github](https://github.com/microsoft/UniVL)  
bert-base-chinese [github](https://github.com/huggingface/transformers)  
注意，有时候代码可能因为http connection请求出错导致中断，此时需要重新运行一次即可

## 比赛心得
1. 特征提取：通过这次比赛我们发现，对于原始数据提取的特征质量直接影响最终预测结果，所以我们对视频特征提取使用了不同的预训练模型来进行特征提取，所使用的模型：1.X3D 2.TSM 3.timesformer 4.VIT，最终敲定VIT模型作为视频特征提取backbone。涨点：大约2个点 
2. 对输入的数据进行随机mask：为了提升模型的鲁棒性，我们对输入的文本、音频和视频特征都进行了随机mask，其中文本部分将输入的字所对应的id进行随机mask，音频和视频分别随机对部分帧进行mask。涨点：大约1个多点  
3. 模型晚期融合：由于不同模态对标签预测有偏执，所以我们采用晚期融合和中间融合相结合的策略，不仅在融合模态层进行预测，各个单独模态都进行预测，最终加权输出标签概率值。涨点：大约两个多点
4. 不要停止预训练：预训练任务能够提高模型的泛化性，并且由于预训练网络规模更大，因此所寻找的函数簇也更多，所以我们在比赛数据集继续进行二次预训练，预训练任务分别是，预测被mask的文字和被mask的视频帧id和音频帧id。涨点：两个多点
5. 标签不平衡：由于这次比赛标签数量长尾分布问题比较严重，因此为了缓解该现象，我们采用了多标签的FocalLoss来缓解标签不平衡所带来的问题。涨点：3个多点  
6. 统一的Transformer架构：为了更好的让模态内和模态间信息更加充分的融合，我们放弃baseline的通道注意力机制，采用统一的transformer架构，self-attention可以更好的捕捉到模态内部以及模态间的关系。涨点：比baseline高出5个点
7. 一些别的trick：在bert-embedding层添加扰动，学习率预热，权重衰减，冻结部分层数、k折融合等。(k折融合yyds！可以在最后冲榜时涨一波大点)。涨点：5折大约3个点，再多融合效果提升不大，其他trick加起来有一个多点
8. 好的预训练模型选择也很重要，但也有队伍不使用过多的预训练模型，仅仅依靠特征就排进了前五，也是非常的厉害。
9. 最终成绩：81.4293

## 这次比赛还可以优化的方向
1. 探索标签的关系：对于最终多标签的预测，我们只是简单的使用了全连接层加sigmoid的策略来进行预测，没有很好的探索标签与标签之间关系，未来会考虑进行层级多标签分类或者label embedding的办法，更好的学习标签与标签之间的关系  
2. 模态融合方式：我们的多模态融合方式为将不同模态concate在一起进行融合，可能会带来模态冗余问题，因此探索更好的模型架构，减缓模态冗余问题。
3. 特征提取部分：可以选择使用更好的特征提取器来进行特征提取，并且可以对特征提取器也进行预训练，从而达到更好的效果。
4. 端到端：直接输入原始视频、音频、文本进行端到端的训练

### 非常感谢这次比赛举办，让我认识了不少大佬，学习了很多东西，收获颇多，希望在多模态方面能和大佬们多多交流。

