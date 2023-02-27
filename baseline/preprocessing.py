import os
import shutil
import cv2
import tqdm
from src import common_func as c_f
from src import face_detector



def create_tmp(data_folder, video_id, del_temp, save_aligned):  # for faster debugging
    """
    Extract frames, detect face, detect facial landmarks, save landmarks.
    --> for debug purpose and to reduce computing time when running multiple models.

    :return: A list l with length = number of frames in the input video.
            l[i] = 0 if no face is detected (means the frame have no information, thus it's real)
            l[i] = -1 if there is a face (means no label, these frames will be assigned predicted label at later stage)
    """

    tmp_folder = os.path.join(data_folder, 'tmp_' + video_id)
    tmp_aligned = os.path.join(data_folder, 'tmp_' + video_id + '_aligned')
    c_f.create_folder(tmp_folder); c_f.create_folder(tmp_aligned)

    print('Extracting frames...')
    ext_frame_cmd = 'ffmpeg -loglevel panic -i ' + os.path.join(data_folder, video_id + '.mp4') + ' ' + tmp_folder + '/f%05d.jpg'
    os.system(ext_frame_cmd)

    print('Detecting landmarks...')
    pretrain_lmk = os.path.join(data_folder, 'shape_predictor_68_face_landmarks.dat')
    # count = 0
    for frame in tqdm.tqdm([i for i in sorted(os.listdir(tmp_folder)) if i.endswith('.jpg')]):
        # if count > 800:
        #     break
        # count += 1
        img = cv2.imread(os.path.join(tmp_folder, frame))
        org_landmarks, aligned_face, aligned_landmarks = face_detector.landmarks_per_frame(img, pretrain_lmk, aligned_type='resize', output_height=768)

        if aligned_landmarks is None:
            continue
        if save_aligned:
            cv2.imwrite(os.path.join(tmp_aligned, frame), aligned_face)
        with open(os.path.join(tmp_folder, 'landmarks.csv'), 'a') as f:
            org_landmarks = [frame.split('.')[0]] + [str(i) for i in org_landmarks.flatten()]
            f.write(','.join(org_landmarks) + '\n')
        with open(os.path.join(tmp_aligned, 'landmarks.csv'), 'a') as f:
            aligned_landmarks = [frame.split('.')[0]] + [str(i) for i in aligned_landmarks.flatten()]
            f.write(','.join(aligned_landmarks) + '\n')

    # finishing, delete tmp files and folders
    if del_temp:
        shutil.rmtree(tmp_folder)


def create_tmp_no_landmarks(data_folder, video_id, del_temp, save_aligned):  # for faster debugging
    """
    Extract frames and detect face.
    --> for debug purpose and to reduce computing time when running multiple models.

    :return: A list l with length = number of frames in the input video.
            tmp_l[i] = 0 if no face is detected (means the frame have no information, thus it's real)
            tmp_l[i] = -1 if there is a face (means no label, these frames will be assigned predicted label at later stage)
    """
    tmp_folder = os.path.join(data_folder, 'tmp_' + video_id)
    tmp_aligned = os.path.join(data_folder, 'tmp_' + video_id + '_aligned_no_landmarks')
    c_f.create_folder(tmp_folder); c_f.create_folder(tmp_aligned)

    print('Extracting frames...')
    ext_frame_cmd = 'ffmpeg -loglevel panic -i ' + os.path.join(data_folder, video_id + '.mp4') + ' ' + tmp_folder + '/f%05d.jpg'
    os.system(ext_frame_cmd)

    print('Detecting faces...')
    # count = 0
    for frame in tqdm.tqdm([i for i in sorted(os.listdir(tmp_folder)) if i.endswith('.jpg')]):
        # if count > 800:
        #     break
        # count += 1
        img = cv2.imread(os.path.join(tmp_folder, frame))
        aligned_face = face_detector.face_per_frame(img, output_height=768)

        if aligned_face is None:
            continue
        if save_aligned:
            cv2.imwrite(os.path.join(tmp_aligned, frame), aligned_face)

    # finishing, delete tmp files and folders
    if del_temp:
        shutil.rmtree(tmp_folder)

