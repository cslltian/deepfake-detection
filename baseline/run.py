"""
python run.py -kt <True/False> -mode <best/all>

-   Video (.mp4) should be put in folder data/ and should include only 1 POI talking in the video.
-   Ground truth should be put in data/, with file name ground_truth.csv.
    Format: video_id, first_fake_frame, last_fake_frame, total_number_of_frames(optional).
        If there is no ground truth, the system will output predicted labels per frame of the input video.
        If there is ground truth, the system will output the Jaccard similarity between predicted and ground truth label.
-   A temporary folder of aligned face extracted from the video will be created and will be deleted at the end.
    Pass True after -kt to keep the temporary folder.
-   This program will run all three baseline models on the input video, with different parameters for each model.
    Three baseline models are based on following artefacts:
    1. Histogram
    2. Luminance
    3. Detailedness

-   The program uses dlib for face detection and landmarks detection.
    Thus, a pre-trained dlib model named shape_predictor_68_face_landmarks.dat should be in the data folder.
-   All faces will be resized to the same height of 768 before putting into the models.

-   Accuracy of using (2) and (3) features will be slightly lower than reported without pre-processing the video.
    To re-produce the same results as report, please remove all frames in the video without a face.
"""
import os
import argparse
import numpy as np
from preprocessing import create_tmp_no_landmarks
import baseline_test as bt
from src import common_func as c_f



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run testing on 1 video. Put the video in the data folder.')
    parser.add_argument('-kt', help='Keeping the temporary folder of aligned faces? (True/False). Default is True.')
    parser.add_argument('-mode', help='Running with all available parameters? (best/all). Default is all.')
    parser.add_argument('-sf', help='Index of first fake frame.', required=True)
    parser.add_argument('-ef', help='Index of last fake frame.', required=True)
    args = parser.parse_args()

    # Preprocessing: detect face, detect landmarks, save to temporary files and folder
    data_folder = 'data'
    videos = [i for i in os.listdir(data_folder) if i.endswith('.mp4')]
    if len(videos) > 1:
        exit(0)
    else:
        video_id = videos[0].split('.mp4')[0]
    tmp_l = create_tmp_no_landmarks(data_folder=data_folder, video_id=video_id, del_temp=False, save_aligned=True)

    tmp_folder = os.path.join(data_folder, 'tmp_' + video_id)
    tmp_aligned = os.path.join(data_folder, 'tmp_' + video_id + '_aligned')

    # Prepare ground truth
    all_frames = sorted(os.listdir(tmp_folder))
    n_frames = len(all_frames)

    # Getting index of frames with faces
    frame_with_face = os.listdir(tmp_aligned)
    gt = c_f.get_ground_truth_label([int(args.sf), int(args.lf)], n_frames)
    non_face_frames = [fid for fid in sorted([i for i in os.listdir(tmp_folder) if i.endswith('.jpg')]) if fid not in frame_with_face]

    # Test histogram features
    # iou_hist = bt.test_hist(tmp_aligned, all_frames, non_face_frames, gt, para_mode='all', save_fig=None)
    # print(iou_hist)

    # Test luminance features
    # iou_lum = bt.test_lum(tmp_aligned, all_frames, non_face_frames, gt, para_mode='all', save_fig=None)
    # print(iou_lum)

    # Test detailedness features
    # Output is a dictionary, with key is in format: <smoothing_window>-<median_range>, and value is the IoU
    # iou_det_face = bt.test_detailedness(tmp_aligned, all_frames, non_face_frames, gt, para_mode='all', roi='face', save_fig=None)
    # iou_det_cheeks = bt.test_detailedness(tmp_aligned, all_frames, non_face_frames, gt, para_mode='all', roi='cheeks', save_fig=None)
    # iou_det_fhcheeks = bt.test_detailedness(tmp_aligned, all_frames, non_face_frames, gt, para_mode='all', roi='fh-cheeks', save_fig=None)
    # print(iou_det_face)
    # print(iou_det_cheeks)
    # print(iou_det_fhcheeks)



