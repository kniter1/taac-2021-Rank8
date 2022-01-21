ROOT='/home/tione/notebook'
DATATYPE="tagging"
VIDEO_PATH=${ROOT}/algo-2021/dataset/videos/video_5k/train_5k
LABEL_PATH=${ROOT}/algo-2021/dataset/label_id.txt
LABEL_INFO_PATH=${ROOT}/algo-2021/dataset/tagging/GroundTruth/tagging_info.txt
VIDEO_FEATURES_PATH=${ROOT}/taac-2021-神奈川冲浪里/pre/VIT_L_train_5k_features
AUDIO_FEATURES_PATH=${ROOT}/algo-2021/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging
VIDEO_CAPTION_PATH=${ROOT}/algo-2021/dataset/tagging/tagging_dataset_train_5k/text_txt/tagging
INIT_MODEL=${ROOT}/taac-2021-神奈川冲浪里/pretrained_models/UniVL_Pretrained_models/pytorch_model.bin.pretrain
OUTPUT_ROOT="ckpts"


python3 -m torch.distributed.launch --nproc_per_node=1 \
train.py \
--do_train --num_thread_reader=12 \
--epochs=15 --batch_size=16 \
--n_display=10 \
--video_path ${VIDEO_PATH} \
--label_path ${LABEL_PATH} \
--label_info_path ${LABEL_INFO_PATH} \
--video_features_path ${VIDEO_FEATURES_PATH} \
--audio_features_path ${AUDIO_FEATURES_PATH} \
--video_caption_path  ${VIDEO_CAPTION_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt --bert_model bert-base-chinese \
--do_lower_case --lr 3e-5 --max_words 200 --max_frames 100 --max_sequence 100 \
--visual_num_hidden_layers 6 \
--audio_num_hidden_layers 3 \
--cache_dir /home/tione/notebook/cmy/tione/notebook/univl/tagging_unvil/tagging/UniVL/cache \
--datatype ${DATATYPE} --init_model ${INIT_MODEL} \
--k_fold 5


