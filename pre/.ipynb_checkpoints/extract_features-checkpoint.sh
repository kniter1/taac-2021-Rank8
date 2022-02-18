ROOT='/home/tione/notebook'
VIDEO_FRAME_PATH=${ROOT}/taac-2021-神奈川冲浪里/pre/test_5k_2nd_frame_npy
SAVE_PATH=${ROOT}/taac-2021-神奈川冲浪里/pre/VTI_L_test_5k_2nd_features
python extract_features.py --frame ${VIDEO_FRAME_PATH} --save ${SAVE_PATH}