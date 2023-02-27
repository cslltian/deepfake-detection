import os
import numpy as np
import cv2
import pandas as pd
import tqdm
from itertools import groupby, product
from matplotlib import pyplot as plt
from src import common_func as c_f
from src.metric import jaccard_similarity
from src import features
import argparse
from src.common_func import smooth, get_ground_truth_label



def get_hist_feature(aligned_folder, save_fig):
    """
    Histogram is computed over the face region (using 68 facial landmarks).
    Difference in histogram between 2 frames is computed using CV_COMP_CORREL.
    Feature vector is normalized into [0, 1] before returning.

    :param aligned_folder: Folder of aligned facial images
    :param save_fig: True to save histogram plot, else False

    :return: Histogram feature of the video
    """
    saved_landmarks_file = os.path.join(aligned_folder, 'landmarks.csv')
    frame_list_to_use = sorted([i for i in os.listdir(aligned_folder) if i.endswith('.jpg')])
    lmks = pd.read_csv(saved_landmarks_file, header=None, index_col=0)

    # read all frame in list, compute histogram
    face_hist = []
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists

    count = 0
    for file in tqdm.tqdm(frame_list_to_use, disable=True):
        # if count > 100:
        #     break
        # count += 1
        img_id = file.split('.')[0]
        img = cv2.imread(os.path.join(aligned_folder, file))
        lmk_array = np.array(lmks.loc[img_id]).reshape(68, 2)

        # Get outer face mask
        face_mask = features.get_face_region_mask(img, lmk_array)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_hist = cv2.calcHist([img_hsv], [0, 1], face_mask, [50, 60], ranges)
        cv2.normalize(img_hist, img_hist, 0, 255, cv2.NORM_MINMAX)
        face_hist.append(img_hist)

    face_hist = np.array(face_hist)
    face_hist_diff = features.get_hist_differences(face_hist)

    # Normalize
    face_hist_diff = (face_hist_diff - min(face_hist_diff)) / (max(face_hist_diff) - min(face_hist_diff))

    # Save plots if save_fig is not None
    if save_fig:
        pass
        # Plotting
        # plot_hist(face_hist_diff, hist_dist_name, norm, smooth_window, savefig)

    return face_hist_diff


def get_lum_feature(aligned_folder, savefig):
    """
    Assumed that a tmp folder includes aligned faces is already generated.
    Images are converted into L*a*b, then L is used for comparing luminance.
    :return:
    """
    frame_list_to_use = sorted([i for i in os.listdir(aligned_folder) if i.endswith('.jpg')])

    # Read all frame in list, compute feature and plot
    video_luminance = []
    count = 0
    for file in tqdm.tqdm(frame_list_to_use):
        # if count > 500:
        #     break
        # count += 1
        img = cv2.imread(os.path.join(aligned_folder, file))
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        luminance = np.sum(img_lab[:, :, 0])
        video_luminance.append(luminance)
    video_luminance = np.array(video_luminance)

    # Replicate first 30 and last 30 with the same value
    video_luminance[:30] = np.repeat(video_luminance[30:31], 30, axis=0)
    video_luminance[-30:] = np.repeat(video_luminance[-30:-29], 30, axis=0)

    return video_luminance


def get_cheeks_detailedness(aligned_folder, savefig):
    saved_landmarks_file = os.path.join(aligned_folder, 'landmarks.csv')
    frame_list_to_use = sorted([i for i in os.listdir(aligned_folder) if i.endswith('.jpg')])
    lmks = pd.read_csv(saved_landmarks_file, header=None, index_col=0)

    # Read all frame in list, compute feature and plot. Features: left cheek, right cheek, forehead
    cheeks_simple = []
    for file in tqdm.tqdm(frame_list_to_use):

        img_id = file.split('.')[0]
        img = cv2.imread(os.path.join(aligned_folder, file))
        lmk_array = np.array(lmks.loc[img_id]).reshape(68, 2)

        # Get cheeks and forehead regions from landmark
        left_cheek_simple, right_cheek_simple = features.get_cheeks_simple(lmk_array)

        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
        img_decomp = features.get_single_wavelet_decomposition(img_gray)

        # combine 3 masks
        frame_feature = []
        for item in img_decomp:
            left_cheek_simple_mask = c_f.mask_from_rect(item.shape, c_f.haar_rect(left_cheek_simple))
            right_cheek_simple_mask = c_f.mask_from_rect(item.shape, c_f.haar_rect(right_cheek_simple))
            combined_mask = cv2.bitwise_or(left_cheek_simple_mask, right_cheek_simple_mask)
            frame_feature.append(np.mean(abs(cv2.bitwise_and(item, item, mask=combined_mask))))

        cheeks_simple.append(frame_feature)

    cheeks_simple = np.array(cheeks_simple)

    # Replicate first 30 and last 30 with the same value
    cheeks_simple[:30, :] = np.repeat(cheeks_simple[30, :].reshape(1, -1), 30, axis=0)
    cheeks_simple[-30:, :] = np.repeat(cheeks_simple[-30, :].reshape(1, -1), 30, axis=0)

    return cheeks_simple


def get_forehead_cheeks_detailedness(aligned_folder, savefig):
    saved_landmarks_file = os.path.join(aligned_folder, 'landmarks.csv')
    frame_list_to_use = sorted([i for i in os.listdir(aligned_folder) if i.endswith('.jpg')])
    lmks = pd.read_csv(saved_landmarks_file, header=None, index_col=0)

    # Read all frame in list, compute feature and plot. Features: left cheek, right cheek, forehead
    fh_cheeks_simple = []
    for file in tqdm.tqdm(frame_list_to_use):
        img_id = file.split('.')[0]
        img = cv2.imread(os.path.join(aligned_folder, file))
        lmk_array = np.array(lmks.loc[img_id]).reshape(68, 2)

        # Get cheeks and forehead regions from landmark
        left_cheek_simple, right_cheek_simple = features.get_cheeks_simple(lmk_array)
        forehead = features.get_forehead(lmk_array)

        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
        img_decomp = features.get_single_wavelet_decomposition(img_gray)

        # combine 3 mask
        frame_feature = []
        for item in img_decomp:
            fh_mask = c_f.mask_from_rect(item.shape, c_f.haar_rect(forehead))

            left_cheek_simple_mask = c_f.mask_from_rect(item.shape, c_f.haar_rect(left_cheek_simple))
            right_cheek_simple_mask = c_f.mask_from_rect(item.shape, c_f.haar_rect(right_cheek_simple))
            combine_simple_mask = cv2.bitwise_or(cv2.bitwise_or(left_cheek_simple_mask, right_cheek_simple_mask), fh_mask)
            # simple feature
            frame_feature.append(np.mean(abs(cv2.bitwise_and(item, item, mask=combine_simple_mask))))

        fh_cheeks_simple.append(frame_feature)

    fh_cheeks_simple = np.array(fh_cheeks_simple)

    # Replicate first 30 and last 30 with the same value
    fh_cheeks_simple[:30, :] = np.repeat(fh_cheeks_simple[30, :].reshape(1, -1), 30, axis=0)
    fh_cheeks_simple[-30:, :] = np.repeat(fh_cheeks_simple[-30, :].reshape(1, -1), 30, axis=0)

    return fh_cheeks_simple


def get_face_detailedness(aligned_folder, save_fig):
    frame_list_to_use = sorted([i for i in os.listdir(aligned_folder) if i.endswith('.jpg')])

    face_noise = []
    for file in tqdm.tqdm(frame_list_to_use):
        img = cv2.imread(os.path.join(aligned_folder, file))

        img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) / 255
        img_decomp = features.get_single_wavelet_decomposition(img_gray)

        frame_feature = []
        for item in img_decomp:
            frame_feature.append(np.mean(abs(item)))
        face_noise.append(frame_feature)

    face_noise = np.array(face_noise)
    # Replicate first 30 and last 30 with the same value
    face_noise[:30, :] = np.repeat(face_noise[30, :].reshape(1, -1), 30, axis=0)
    face_noise[-30:, :] = np.repeat(face_noise[-30, :].reshape(1, -1), 30, axis=0)

    return face_noise


def predict_with_median(face_frames, noises, threshold):
    noise_median = np.median(noises)
    upper = noise_median + threshold
    lower = noise_median - threshold

    fake_frame_idx = []
    for idx, v in enumerate(face_frames):
        if not (lower < noises[idx] < upper):
            fake_frame_idx.append(v)
    return fake_frame_idx


def test_hist(input_face_folder, all_frames, non_face_id, gt_label, para_mode, save_fig):
    """
    Test histogram features. Run both scenario.
    :param: para_mode: Run all available parameter if 'all', else run the 'best'.
                       'all' is with threshold t in [0.5, 0.6, 0.7, 0.8, 0.9]
                       'best' is when threshold t is 0.7.
    :param: save_fig: if not None, save plots of histogram.
    :return: IoU of 2 scenario.
    """
    print('FEATURE TYPE: Histogram')
    if para_mode == 'all':
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        thresholds = [0.7]
    iou_dict = {}

    f_hist = get_hist_feature(input_face_folder, save_fig)
    f_hist_diff = (f_hist - min(f_hist)) / (max(f_hist) - min(f_hist))

    # Run with 2 scenarios: one fake segment AND one-or-more fake segment
    for corr_threshold in thresholds:
        print('HISTOGRAM THRESHOLD =', corr_threshold)
        # Find CHANGE from real to fake frames based on correlation threshold
        # skip some first and last frames
        # f_hist_diff is bw 2 consecutive frames, thus, assuming last frame is real
        s_skip, e_skip = 10, 10
        peaks = [0] * s_skip + [int(item < corr_threshold) for item in f_hist_diff[s_skip:-e_skip]] + [0] * (e_skip + 1)
        peaks_idx = [idx for idx, v in enumerate(peaks) if v == 1]

        if len(peaks_idx) >= 2:
            # print(peaks_idx)
            if len(peaks_idx) % 2 == 1:
                print('Remove last bump!')
                peaks_idx = peaks_idx[:-1]

            # PREDICTION based on bump_idx
            # first bump is from real to fake, e.g. from 200 to 201, if bump[200] = 1 means 201 is fake
            # second bump is from fake to real, e.g. from 675 to 676 => 675 is last fake

            face_frames = [i for i in sorted(os.listdir(input_face_folder)) if i.endswith('.jpg')]

            # One fake segment
            fake_frame_idx_first = face_frames[peaks_idx[0]+1: peaks_idx[1]+1]
            prediction_first_scenario = [(idx not in non_face_id and idx in fake_frame_idx_first) for idx in all_frames]
            iou_first_scenario = jaccard_similarity(prediction_first_scenario, gt_label)[-1]

            # One-or-more fake segment
            bumps_idx_pair = np.reshape(peaks_idx, (int(len(peaks_idx)/2), 2))

            # prediction_second_scenario = [0] * len(peaks)
            fake_frame_idx_second = []
            for s, e in bumps_idx_pair:
                fake_frame_idx_second.extend(face_frames[s+1:e+1])
            prediction_second_scenario = [(idx not in non_face_id and idx in fake_frame_idx_second) for idx in all_frames]
            iou_second_scenario = jaccard_similarity(prediction_second_scenario, gt_label)[-1]

            iou_dict[corr_threshold] = [iou_first_scenario, iou_second_scenario]
        else:
            print('NO peaks index for', str(corr_threshold))

    return iou_dict


def test_lum(input_face_folder, all_frames, non_face_id, gt_label, para_mode, save_fig):
    print('FEATURE TYPE: Luminance')
    if para_mode == 'all':
        windows = [0, 30, 50, 100]
        ranges = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        windows = [100]
        ranges = [0.35]
    iou_dict = {}

    video_luminance = get_lum_feature(input_face_folder, save_fig)
    for w in windows:
        for r in ranges:
            iou_key = str(w) + '-' + str(r)
            print('LUMINANCE: w=', w, 'r=', r)

            if w != 0:  # = 0 means no smoothing
                video_luminance = smooth(video_luminance, w)
            else:
                video_luminance = video_luminance
            video_luminance = (video_luminance - min(video_luminance)) / (max(video_luminance) - min(video_luminance))  # Normalize

            face_frames = [i for i in sorted(os.listdir(input_face_folder)) if i.endswith('.jpg')]

            fake_frame_idx = predict_with_median(face_frames, video_luminance, r)
            predicted_labels = [(idx not in non_face_id and idx in fake_frame_idx) for idx in all_frames]
            iou_dict[iou_key] = jaccard_similarity(predicted_labels, gt_label)[-1]

    return iou_dict


def test_detailedness(input_face_folder, all_frames, non_face_id, gt_label, para_mode, roi, save_fig):
    print('FEATURE TYPE: Detailedness')
    if para_mode == 'all':
        windows = [0, 30, 50, 100]
        ranges = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    else:
        windows = [100]
        ranges = [0.35]
    iou_dict = {}

    if roi == 'face':
        video_detailedness = get_face_detailedness(input_face_folder, save_fig)
    elif roi == 'cheeks':
        video_detailedness = get_cheeks_detailedness(input_face_folder, save_fig)
    elif roi == 'fh-cheeks':
        video_detailedness = get_forehead_cheeks_detailedness(input_face_folder, save_fig)

    for w in windows:
        for r in ranges:
            iou_key = str(w) + '-' + str(r)
            video_detailedness_cA = video_detailedness[:, 0]

            if w != 0:  # = 0 means no smoothing
                video_detailedness_cA = smooth(video_detailedness_cA, w)
            else:
                video_detailedness_cA = video_detailedness_cA
            video_detailedness_cA = (video_detailedness_cA - min(video_detailedness_cA)) / (max(video_detailedness_cA) - min(video_detailedness_cA))  # Normalize

            face_frames = [i for i in sorted(os.listdir(input_face_folder)) if i.endswith('.jpg')]
            fake_frame_idx = predict_with_median(face_frames, video_detailedness_cA, r)
            predicted_labels = [(idx not in non_face_id and idx in fake_frame_idx) for idx in all_frames]
            iou_dict[iou_key] = jaccard_similarity(predicted_labels, gt_label)[-1]

    return iou_dict
