import numpy as np
import pywt
from src.common_func import FACIAL_LANDMARKS_INDEX as FLI
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.linear_model import LinearRegression
from itertools import product


def get_cheeks_simple(landmark):
    """
    Simple boxes:
        both cheeks have height = dist(lmk_30 to lmk_34);
        width = dist(lmk_5 to lmk_49) for left cheek and dist(lmk_55 to lmk_13) for right cheek
    :param landmark: 68x2 array
    :return: 2 rectangles, each with (x_top_left, y_top_left, x_bottom_right, y_bottom_right) for left cheek and right cheek
    """
    x_range = np.array([30, 34]) - 1
    y_range_left = np.array([5, 49]) - 1
    y_range_right = np.array([55, 13]) - 1

    right_cheek = (landmark[x_range[0], 1], landmark[y_range_right[0], 0], landmark[x_range[1], 1], landmark[y_range_right[1], 0])
    left_cheek = (landmark[x_range[0], 1], landmark[y_range_left[0], 0], landmark[x_range[1], 1], landmark[y_range_left[1], 0])

    return left_cheek, right_cheek


def get_cheeks_complex(landmarks):
    """
    Complicated boxes
    :param landmarks: 68x2 array
    :return: 2 rectangles, each with (x_top_left, y_top_left, x_bottom_right, y_bottom_right) for left cheek and right cheek
    """
    left_center = (4 * landmarks[0] + 2 * landmarks[4] + 3 * landmarks[31]) / 9
    left_width = (landmarks[31, 0] - landmarks[4, 0]) * 0.35
    left_height = (landmarks[4, 1] - landmarks[0, 1]) * 0.35
    left_cheek = (int(left_center[1] - left_height/2), int(left_center[0] - left_width/2), int(left_center[1] + left_height/2), int(left_center[0] + left_width/2))

    right_center = (4 * landmarks[16] + 2 * landmarks[12] + 3 * landmarks[35]) / 9
    right_width = (landmarks[12, 0] - landmarks[35, 0]) * 0.35
    right_height = (landmarks[12, 1] - landmarks[16, 1]) * 0.35
    right_cheek = (int(right_center[1] - right_height/2), int(right_center[0] - right_width/2), int(right_center[1] + right_height/2), int(right_center[0] + right_width/2))

    return left_cheek, right_cheek


def get_forehead(landmarks):
    """
    Forehead is cut from eyebrows to top of image, width = dist(middle of 2 eyebrows)
    :param landmarks: 68x2 array
    :return: rectangles: (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    """
    mid_left_idx = (FLI['left_eyebrow'][0] + FLI['left_eyebrow'][1] - 1) / 2
    mid_right_idx = (FLI['right_eyebrow'][0] + FLI['right_eyebrow'][1] - 1) / 2

    eyebrows = np.vstack((landmarks[FLI['left_eyebrow'][0]:FLI['left_eyebrow'][1], :], landmarks[FLI['right_eyebrow'][0]:FLI['right_eyebrow'][1], :]))

    x_highest = np.min(eyebrows[:, 1])
    forehead_top_left = (0, landmarks[int(mid_left_idx)][1])
    forehead_bottom_right = (x_highest - 10, landmarks[int(mid_right_idx)][0])

    return forehead_top_left + forehead_bottom_right


def get_single_wavelet_decomposition(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    cA, (cH, cV, cD) = coeffs2

    return cA, cH, cV, cD


def get_Daubechies_decomposition(img, db_type):
    # db_type = ['db4', 'db6']
    coeffs2 = pywt.dwt2(img, db_type)
    cA, (cH, cV, cD) = coeffs2

    return cA, cH, cV, cD


def get_eye_measurements(landmarks, side):
    """

    :param landmarks:
    :param side: left_eye or right_eye
    :return:
    """
    eye_landmarks = landmarks[FLI[side][0]:FLI[side][1], :]

    width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    height1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    height2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # height = height1 if height1 > height2 else height2
    height = (height1 + height2) / 2

    # hull = cv2.convexHull(np.array(eye_landmarks, dtype=np.float32))

    return width, height, width * height


def get_eye_aspect_ratio(landmarks, side):
    eye_landmarks = landmarks[FLI[side][0]:FLI[side][1], :]

    width = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    height1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    height2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    return (height1 + height2) / (2 * width)


def get_avg_color(img, mask):
    """
    Get average color of region(s) of an image based on a binary mask
    :param img:
    :param mask:
    :return:
    """
    sl_x, sl_y = np.where(mask)
    selected_region = np.array([img[x, y, :] for x, y in zip(sl_x, sl_y)])
    avg_color = np.mean(selected_region, axis=0)

    avg_color_img = (np.ones(img.shape) * avg_color).astype('uint8')
    # debug = np.zeros(img.shape, dtype='uint8')
    # for x, y in zip(sl_x, sl_y):
    #     debug[x, y, :] = img[x, y, :]
    #
    # plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('img')
    # plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)), plt.title('selected region')
    # plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(avg_color_img, cv2.COLOR_BGR2RGB)), plt.title('average color')
    # plt.show()
    return avg_color, avg_color_img


def get_color_difference(colors):
    """
    Compute difference between every 2 consecutive colors.
    Distance: deltaE: https://en.wikipedia.org/wiki/Color_difference
    :param colors: Nx3 array, with N = number of color elements. Colors are in L*a*b.
    :return: list with length = N-1
    """
    differences = []
    for idx in range(colors.shape[0] - 1):
        deltaE = np.linalg.norm(colors[idx, :] - colors[idx + 1, :])
        differences.append(deltaE)

    return np.array(differences)


def get_avg_color_differences(img1, img2, mask1, mask2):
    """
    Given 2 images, compare difference between colors at masked regions. Images are same shape.
    Return average differences
    :param img1: image 1 - MxNxD
    :param img2: image 2 - MxNxD
    :return:
    """
    # Convert images to L*a*b
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2Lab)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2Lab)

    # Get masked regions
    sl_x1, sl_y1 = np.where(mask1)
    sl_x2, sl_y2 = np.where(mask2)
    selected_region_1 = np.array([img1[x, y, :] for x, y in zip(sl_x1, sl_y1)])
    selected_region_2 = np.array([img2[x, y, :] for x, y in zip(sl_x2, sl_y2)])
    print(selected_region_1.shape, selected_region_2.shape)

    assert selected_region_1.shape == selected_region_2.shape

    # Compute color differences between 2 images and take the average
    differences = []
    for idx in range(selected_region_1.shape[0]):
        deltaE = np.linalg.norm(selected_region_1[idx, :] - selected_region_2[idx, :])
        differences.append(deltaE)
    avg_diff = np.mean(np.array(differences))

    return avg_diff


def get_dominant_color(img, mask):
    """
    Get dominant color of the masked region(s) in img
    :param img:
    :param mask:
    :return:
    """
    # Convert images to L*a*b
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Get masked regions
    sl_x, sl_y = np.where(mask)
    selected_region = np.array([img[x, y, :] for x, y in zip(sl_x, sl_y)], dtype=np.float32)

    # Choose dominant color using kmeans
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(selected_region, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    # dominant_img = cv2.cvtColor((np.ones(img.shape) * dominant).astype('uint8'), cv2.COLOR_Lab2BGR)

    return dominant


def get_global_lighting(img):
    """
    Given an image, take 1 channel, e.g. Red channel.
    Compute global lighting average plane and return surface normal.
    :param img: in BGR
    :return:
    """
    red_value = img[:, :, 2].flatten()
    xs = range(img.shape[0])
    ys = range(img.shape[1])
    indices = np.array(list(product(xs, ys)))

    # plotting
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(indices[:, 0], indices[:, 1], red_value, label='parametric curve')
    # ax.legend()
    # plt.show()
    # plotting

    # Get average plane
    reg = LinearRegression().fit(indices, red_value)
    norm_vect = reg.coef_
    norm_vect_normalized = norm_vect / np.sqrt(np.sum(norm_vect ** 2))

    return norm_vect_normalized


def get_surface_normal_difference(normals):
    """
    Compute difference between every 2 consecutive normal vectors.
    :param normals:
    """
    differences = []
    for idx in range(normals.shape[0] - 1):
        cos_alpha = np.dot(normals[idx], normals[idx+1])
        differences.append(cos_alpha)

    return np.array(differences)


def get_lightness(img, mask):
    """
    Transform to L*a*b
    :param img: image in BGR
    :param mask:
    :return:
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lightness = img_lab[:, :, 0]
    lightness_100 = (lightness / 255) * 100

    sl_x, sl_y = np.where(mask)
    selected_region_lightness = np.array([lightness_100[x, y] for x, y in zip(sl_x, sl_y)])
    selected_region_lightness_avg = np.mean(selected_region_lightness)

    return selected_region_lightness_avg


def get_face_region_mask(image, landmarks):
    """
    :param image:
    :param landmarks:
    :return: mask in numpy darray and in np.uint8 type
    """

    outer_face_landmarks = landmarks[FLI['outer_face'][0]:FLI['outer_face'][1], :]

    hull = cv2.convexHull(np.array(outer_face_landmarks, dtype=np.float32))
    mask = np.zeros(image.shape, np.uint8)
    mask = cv2.drawContours(mask, [hull.astype(int)], 0, (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 2, 2), plt.imshow(mask, cmap='gray')
    # plt.show()
    return mask


def get_hist_differences(hists):
    """
    Comparing histogram using cv.HISTCMP_CORREL
    :param hists:
    :return:
    """

    differences = []
    for idx in range(hists.shape[0] - 1):
        d = cv2.compareHist(hists[idx], hists[idx+1], method=cv2.HISTCMP_CORREL)

        differences.append(d)

    return np.array(differences)

