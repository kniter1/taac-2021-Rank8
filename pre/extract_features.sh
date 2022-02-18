ROOT='/home/tione/notebook'
VIDEO_FRAME_PATH=${ROOT}/taac-2021-S/pre/test_2nd_5k_384_frame_npy
SAVE_PATH=${ROOT}/taac-2021-S/pre/VTI_L_test_5k_2nd_features
python extract_features.py --frame ${VIDEO_FRAME_PATH} --save ${SAVE_PATH}