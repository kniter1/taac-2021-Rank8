ROOT='/home/tione/notebook'
VIDEO_PATH=${ROOT}/algo-2021/dataset/videos/test_5k_2nd
SAVE_PATH=${ROOT}/taac-2021-神奈川冲浪里/pre/test_2nd_5k_384_frame_npy
python extract_video_frame.py --video ${VIDEO_PATH} --save ${SAVE_PATH}