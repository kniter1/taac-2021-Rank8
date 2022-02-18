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
然后由于特征文件是lfs上传的所以还需要下载lfs，输入以下指令下载lfs：
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash  
sudo apt-get install git-lfs
```
然后输入
```
git lfs pull
```
之后才能获得正常特征文件
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

>>>>>>> master
