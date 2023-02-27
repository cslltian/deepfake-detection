from collections import OrderedDict
import numpy as np
import os
import shutil



# Index start from 0, inclusive of last element, so that when we call it, can simply use [start:end]
FACIAL_LANDMARKS_INDEX = OrderedDict([
    ("outer_face", (0, 27)),
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("left_eyebrow", (17, 22)),
    ("right_eyebrow", (22, 27)),
    ("left_eye", (36, 42)),
    ("right_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])



def landmarks_shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # Convert a bounding from dlib output to (x, y, w, h) in OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    elif input(folder_path + " is existed! Overwrite? (Y/N) ").lower() == 'y':
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
    else:
        print('Handle folder', folder_path, 'manually!!!')
        exit(0)


def check_nan(check_list):
    for item in check_list:
        if item.lowercase() == 'nan':
            return True
    return False


def mask_from_rect(mask_shape, rect):
    """
    Given (x,y) position of top_left and bottom_right points, return a mask with mask_shape and the rectangle region is equal 1
    :param mask_shape: (height, width)
    :param rect: (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    :return: mask
    """
    mask = np.zeros(mask_shape, dtype='uint8')
    mask[rect[0]:rect[2], rect[1]:rect[3]] = 255

    return mask


def haar_rect(rect):
    """

    :param rect: (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    :return:
    """
    h, w = int((rect[2] - rect[0])/2), int((rect[3] - rect[1])/2)
    return int(rect[0]/2), int(rect[1]/2), int(rect[0]/2) + h, int(rect[1]/2) + w


def smooth(y, box_pts):
    """
    To avoid boundary effect, replicate the first and last element.
    Then after convolve, remove the first and last element
    :param y:
    :param box_pts:
    :return:
    """
    box = np.ones(box_pts) / box_pts
    y_list = y.tolist()
    y = [y[0]] * int(box_pts/2) + y_list + [y[-1]] * (int(box_pts/2) - 1)
    # y = np.insert(y, [0], y[0])
    # y = np.insert(y, [y.size], y[-1])
    y = np.array(y)
    y_smooth = np.convolve(y, box, mode='valid')
    # y_smooth = y_smooth[int(box_pts/2):-int(box_pts/2)+1]
    # print(y.shape, y_smooth.shape)
    return y_smooth


def get_ground_truth_label(gt_info, total_frames):
    """
    :param gt_info: first_fake, last_fake, total (optional)
    :return:
    """
    if len(gt_info) == 2:  # no total
        first_fake, last_fake = gt_info
    else:
        first_fake, last_fake, _ = gt_info
    ground_truth = [0] * total_frames
    fake_length = last_fake - first_fake + 1
    ground_truth[first_fake:(last_fake + 1)] = [1] * fake_length

    return ground_truth

