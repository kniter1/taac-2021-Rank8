ROOT='/home/tione/notebook'
DATATYPE="tagging"
VIDEO_PATH=${ROOT}/total_video
VIDEO_FEATURES_PATH=${ROOT}/VIL_L_total_features
AUDIO_FEATURES_PATH=${ROOT}/total_audio_npy
VIDEO_CAPTION_PATH=${ROOT}/total_text
INIT_MODEL=${ROOT}/cmy/tione/notebook/univl/tagging_unvil/tagging/UniVL/weights/univl.pretrained.bin
BERT_MODEL=${ROOT}/cmy/tione/notebook/univl/tagging_unvil/bert/pytorch_model.bin
OUTPUT_ROOT="ckpts"


python3 -m torch.distributed.launch --nproc_per_node=1  \
main_pretrain.py \
--do_pretrain --num_thread_reader=8 \
--epochs=50 --batch_size=10 \
--n_display=10 \
--video_path ${VIDEO_PATH} \
--video_features_path ${VIDEO_FEATURES_PATH} \
--audio_features_path ${AUDIO_FEATURES_PATH} \
--video_caption_path  ${VIDEO_CAPTION_PATH} \
--output_dir ${OUTPUT_ROOT}/ckpt/pretrain_3nd --bert_model bert-base-chinese \
--lr 5e-5 --max_words 400 --max_frames 150 --max_sequence 150 \
--visual_num_hidden_layers 6 \
--audio_num_hidden_layers 3 \
--bert_model_path ${BERT_MODEL} \
--cache_dir /home/tione/notebook/cmy/tione/notebook/univl/tagging_unvil/tagging/UniVL/cache \
--datatype ${DATATYPE} --init_model ${INIT_MODEL} 