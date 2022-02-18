ROOT=/home/tione/notebook
MODEL_FILE=${ROOT}/taac-2021-神奈川冲浪里/ckpts/ckpt
OUTPUT_JSON=${ROOT}/taac-2021-神奈川冲浪里
VIDEO_PATH=${ROOT}/algo-2021/dataset/videos/test_5k_2nd
LABEL_PATH=${ROOT}/algo-2021/dataset/label_id.txt
VIDEO_FEATURES_PATH=${ROOT}/VIT_L_test_5k_2nd_features
AUDIO_FEATURES_PATH=${ROOT}/algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd/audio_npy/Vggish/tagging
VIDEO_CAPTION_PATH=${ROOT}/algo-2021/dataset/tagging/tagging_dataset_test_5k_2nd/text_txt/tagging
CACHE_DIR=${ROOT}/cmy/tione/notebook/univl/tagging_unvil/tagging/UniVL/cache

python3 -m inference --model_file ${MODEL_FILE} \
--output_json ${OUTPUT_JSON}/test.json \
--video_path ${VIDEO_PATH} \
--label_path ${LABEL_PATH} \
--video_features_path ${VIDEO_FEATURES_PATH} \
--audio_features_path ${AUDIO_FEATURES_PATH} \
--video_caption_path ${VIDEO_CAPTION_PATH} \
--cache_dir ${CACHE_DIR} 