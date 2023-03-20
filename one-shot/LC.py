import cv2
import numpy as np
from collections import OrderedDict
import csv

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


def readin_csv(file):
    with open(file+".csv", 'r') as f:
        reader = csv.reader(f)
        shape_sequence = []
        for i, record in enumerate(reader):
            if i == 0:
                continue
            # if eval(record[4]) != 1:
            #     continue
            landmarks = []
            for j in range(68):
                landmarks.append((eval(record[5 + j]), eval(record[5 + 68 + j])))
            shape_sequence.append(landmarks)
    return shape_sequence


def shape_to_face(shape, width, height, scale=1.2):
    """
    Recalculate the face bounding box based on coarse landmark location(shape)
    :param
    shape: landmark locations
    scale: the scale parameter of face, to enlarge the bounding box
    :return:
    face_new: new bounding box of face (1*4 list [x1, y1, x2, y2])
    # face_center: the center coordinate of face (1*2 list [x_c, y_c])
    face_size: the face is rectangular( width = height = size)(int)
    """
    x_min, y_min = np.min(shape, axis=0)
    x_max, y_max = np.max(shape, axis=0)

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2

    face_size = int(max(x_max - x_min, y_max - y_min) * scale)
    # Enforce it to be even
    # Thus the real whole bounding box size will be a odd
    # But after cropping the face size will become even and
    # keep same to the face_size parameter.
    face_size = face_size // 2 * 2

    x1 = max(x_center - face_size // 2, 0)
    y1 = max(y_center - face_size // 2, 0)

    face_size = min(width - x1, face_size)
    face_size = min(height - y1, face_size)

    x2 = x1 + face_size
    y2 = y1 + face_size

    face_new = [int(x1), int(y1), int(x2), int(y2)]
    return face_new, face_size


def landmark_align(shape):
    desiredLeftEye = (0.35, 0.25)
    desiredFaceWidth=2
    desiredFaceHeight=2
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0)#.astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0)#.astype("int")
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))  # - 180

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                  (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = 0#desiredFaceWidth * 0.5
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    n, d = shape.shape
    temp = np.zeros((n, d + 1), dtype="int")
    temp[:, 0:2] = shape
    temp[:, 2] = 1
    aligned_landmarks = np.matmul(M, temp.T)
    return aligned_landmarks.T #.astype("int"))


def check_and_merge(location, forward, feedback, P_predict, status_fw=None, status_fb=None):
    num_pts = 68
    check = [True] * num_pts

    target = location[1]
    forward_predict = forward[1]

    # To ensure the robustness through feedback-check
    forward_base = forward[0]  # Also equal to location[0]
    feedback_predict = feedback[0]
    feedback_diff = feedback_predict - forward_base
    feedback_dist = np.linalg.norm(feedback_diff, axis=1, keepdims=True)

    # For Kalman Filtering
    detect_diff = location[1] - location[0]
    detect_dist = np.linalg.norm(detect_diff, axis=1, keepdims=True)
    predict_diff = forward[1] - forward[0]
    predict_dist = np.linalg.norm(predict_diff, axis=1, keepdims=True)
    predict_dist[np.where(predict_dist == 0)] = 1  # Avoid nan
    P_detect = (detect_dist / predict_dist).reshape(num_pts)

    for ipt in range(num_pts):
        if feedback_dist[ipt] > 2:  # When use float
            check[ipt] = False

    if status_fw is not None and np.sum(status_fw) != num_pts:
        for ipt in range(num_pts):
            if status_fw[ipt][0] == 0:
                check[ipt] = False
    if status_fw is not None and np.sum(status_fb) != num_pts:
        for ipt in range(num_pts):
            if status_fb[ipt][0] == 0:
                check[ipt] = False
    location_merge = target.copy()
    # Merge the results:
    """
    Use Kalman Filter to combine the calculate result and detect result.
    """

    Q = 0.3  # Process variance

    for ipt in range(num_pts):
        if check[ipt]:
            # Kalman parameter
            P_predict[ipt] += Q
            K = P_predict[ipt] / (P_predict[ipt] + P_detect[ipt])
            location_merge[ipt] = forward_predict[ipt] + K * (target[ipt] - forward_predict[ipt])
            # Update the P_predict by the current K
            P_predict[ipt] = (1 - K) * P_predict[ipt]
    return location_merge, check, P_predict


def calibrate_landmark(frames, shape_sequence):
    frames_num = len(frames)
    shape_num = len(shape_sequence)
    # assert frames and landmarks are corresponding.
    assert frames_num == shape_num
    frame_height, frame_width = frames[0].shape[:2]
    """
    Pre-process:
    To detect the original results,
    and normalize each face to a certain width, 
    also its corresponding landmarks locations and 
    scale parameter.
    """
    face_size_normalized = 400
    faces = []
    locations = []
    shapes_para = []  # Use to recover the shape in whole frame. ([x1, y1, scale_shape])

    for i in range(frames_num):
        frame = frames[i]
        shape = shape_sequence[i]
        face, face_size = shape_to_face(shape, frame_width, frame_height, 1.2)
        faceFrame = frame[face[1]: face[3], face[0]:face[2]]
        if face_size < face_size_normalized:
            inter_para = cv2.INTER_CUBIC
        else:
            inter_para = cv2.INTER_AREA
        face_norm = cv2.resize(faceFrame, (face_size_normalized, face_size_normalized), interpolation=inter_para)
        scale_shape = face_size_normalized/face_size
        shape_norm = np.rint((shape-np.array([face[0], face[1]])) * scale_shape).astype(int)
        faces.append(face_norm)
        shapes_para.append([face[0], face[1], scale_shape])
        # shapes_origin.append(shape)
        locations.append(shape_norm)

    """
    Calibration module.
    """
    segment_length = 2
    locations_sum = len(locations)
    if locations_sum == 0:
        return []
    locations_track = [locations[0]]
    num_pts = 68
    P_predict = np.array([0] * num_pts).reshape(num_pts).astype(float)

    for i in range(locations_sum - 1):
        faces_seg = faces[i:i + segment_length]
        locations_seg = locations[i:i + segment_length]
        lk_params = dict(winSize=(15, 15),
                         maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Use the tracked current location as input. Also use the next frame's predicted location for
        # auxiliary initialization.

        start_pt = locations_track[i].astype(np.float32)
        target_pt = locations_seg[1].astype(np.float32)

        forward_pt, status_fw, err_fw = cv2.calcOpticalFlowPyrLK(faces_seg[0], faces_seg[1],
                                                                 start_pt, target_pt, **lk_params,
                                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        feedback_pt, status_fb, err_fb = cv2.calcOpticalFlowPyrLK(faces_seg[1], faces_seg[0],
                                                                  forward_pt, start_pt, **lk_params,
                                                                  flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        forward_pts = [locations_track[i].copy(), forward_pt]
        feedback_pts = [feedback_pt, forward_pt.copy()]

        forward_pts = np.rint(forward_pts).astype(int)
        feedback_pts = np.rint(feedback_pts).astype(int)

        merge_pt, check, P_predict = check_and_merge(locations_seg, forward_pts, feedback_pts, P_predict, status_fw,
                                                     status_fb)
        locations_track.append(merge_pt)

    """
    Align and normalize the landmark for model training.
    """
    calibrated_aligned_landmarks = []
    for i in locations_track:
        shape = landmark_align(i)

        shape = shape.ravel()
        shape = shape.tolist()
        calibrated_aligned_landmarks.append(shape)

    return calibrated_aligned_landmarks


def calibrator(video_file, landmark_sequence):
    """
    :param video_file: "xxx.mp4" (str)
    :param landmark_sequence: A sequence contains the landmark positions in each frame.
                shape: (frames_num, 68, 2)
    :return: calibrated_aligned_landmarks. shape: (frames_num, 136)
    """
    vidcap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            break
    calibrated_aligned_landmarks = calibrate_landmark(frames, landmark_sequence)
    vidcap.release()
    return np.array(calibrated_aligned_landmarks)


"""
Example code
"""
# video_name = "./example/000_003"
# landmark_sequence = readin_csv(video_name)
# results = calibrator(video_name + '.mp4', landmark_sequence)
# np.savetxt(video_name+".txt", results, fmt='%1.5f')