# Detect face & facial landmarks using dlib
import tqdm
import numpy as np
import cv2
import dlib
from torchvision.io import read_video

from src import common_func as c_f


def get_face_region(img):
    """
    Detect face region in the image. There's only 1 face per image.
    If there is more than 1 face, skip the frame
    """
    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()
    face_rects = detector(img, 1)
    if len(face_rects) != 1:
        # print('Wrong input! Detected multiple faces...')
        return None

    return face_rects[0]


def facial_alignment(img, landmarks, output_shape, desired_left_eye):
    """
    Align faces based on 3 criteria:
    (1) Face is centered
    (2) Eyes horizontal line is straight
    (3) Output faces are the same size with eyes' distances are fixed

    :param img: full input image with only 1 face
    :param landmarks: predicted landmarks of the face
    :param output_shape: (W, H) = (256, 256)
    :param desired_left_eye: (x,y): desired relative position of left eye to the output horizontal length
            x, y are in range of [0 to 1]

    :return: aligned facial image
    """
    # Get left eye, right eye and center of eyes coordinates
    left_start, left_end = c_f.FACIAL_LANDMARKS_INDEX["left_eye"]
    right_start, right_end = c_f.FACIAL_LANDMARKS_INDEX["right_eye"]
    left_eye = landmarks[left_start:left_end]
    right_eye = landmarks[right_start:right_end]
    left_center, right_center = np.mean(left_eye, axis=0), np.mean(right_eye, axis=0)

    # (1) Face is centered => rotation center is eyes' center
    eyes_center = ((left_center[0] + right_center[0]) // 2, (left_center[1] + right_center[1]) // 2)

    # (2) Eyes horizontal line is straight => rotation angle
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    alpha = np.degrees(np.arctan2(dY, dX))

    # (3) Output faces are the same size with eyes' distances are fixed => scale = desired / current_distance
    desired_left_eye_x, desired_left_eye_y = desired_left_eye
    desired_right_eye_x = 1.0 - desired_left_eye_x
    current_dist = np.sqrt((dX ** 2) + (dY ** 2))
    desired_dist = (desired_right_eye_x - desired_left_eye_x) * 256
    scale = desired_dist / current_dist

    # (1) Face is centered => move face to center of output
    tX = output_shape[0] * 0.5 - eyes_center[0]
    tY = output_shape[1] * desired_left_eye_y - eyes_center[1]

    # Combine all criteria to transform matrix M and perform warpAffine
    M = cv2.getRotationMatrix2D(eyes_center, alpha, scale)
    M[0, 2] += tX
    M[1, 2] += tY
    aligned = cv2.warpAffine(img, M, output_shape, flags=cv2.INTER_CUBIC)

    return aligned, M


def landmarks_per_frame(img, pre_trained_landmarks, aligned_type, output_height):
    # detect face and landmark
    lmk_predictor = dlib.shape_predictor(pre_trained_landmarks)
    face_bb = get_face_region(img)

    if face_bb is not None:
        face_lmk = c_f.landmarks_shape_to_np(lmk_predictor(img, face_bb))

        if aligned_type == 'standard':
            aligned_face, _ = facial_alignment(img, face_lmk, (256, 256), desired_left_eye=(0.3, 0.3))

        elif aligned_type == 'resize' and output_height is not None:
            (x, y, w, h) = c_f.rect_to_bb(face_bb)
            face_region = img[y-5:y+w+5, x-5:x+h+5]
            ratio = output_height / h
            dim = (int(w * ratio), int(h * ratio))  # (width, height)
            aligned_face = cv2.resize(face_region, dim)

        # newly detected landmark
        aligned_face_bb = get_face_region(aligned_face)
        if aligned_face_bb is not None:
            aligned_face_lmk = c_f.landmarks_shape_to_np(lmk_predictor(aligned_face, aligned_face_bb))
            return face_lmk, aligned_face, aligned_face_lmk
        else:
            return None, None, None
    else:
        return None, None, None


def landmarks_on_videos(video_path, pre_trained_landmarks, landmarks_csv, aligned_type):
    frames, _, info = read_video(video_path, pts_unit='sec')

    count = 0
    for frame in tqdm.tqdm(frames):
        frame = frame.numpy()
        aligned_lmk = landmarks_per_frame(frame, pre_trained_landmarks, count, aligned_type)
        count += 1

        if aligned_lmk is not None:
            with open(landmarks_csv, 'a') as f:
                csv_rows = ["{},{}".format(i, j) for i, j in aligned_lmk]
                f.write(','.join(csv_rows) + '\n')


def face_per_frame(img, output_height):
    # align type can only be resize
    # detect face and landmark
    face_bb = get_face_region(img)

    if face_bb is not None:
        (x, y, w, h) = c_f.rect_to_bb(face_bb)
        face_region = img[y-5:y+w+5, x-5:x+h+5]
        ratio = output_height / h
        dim = (int(w * ratio), int(h * ratio))  # (width, height)
        aligned_face = cv2.resize(face_region, dim)

        return aligned_face
    else:
        return None


