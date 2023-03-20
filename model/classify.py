import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
tf.get_logger().setLevel('INFO')

block_size = 25
mixed_pred_threshold = 0.4
DROPOUT_RATE = 0.5
RNN_UNIT = 64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
device = "CPU" if len(gpus) == 0 else "GPU"
print("Using device: {}".format(device))


def smooth_predictions(pred):
    out_pred = []
    for i in range(len(pred)):
        prev_lbl = pred[i - 1] if i > 0 else -1
        next_lbl = pred[i + 1] if i < len(pred) - 1 else -1
        if pred[i] == 1 and prev_lbl == 0 and next_lbl == 0:
            out_pred.append(0)
        elif pred[i] == 0 and prev_lbl == 1 and next_lbl == 1:
            out_pred.append(1)
        else:
            out_pred.append(pred[i])
    return out_pred


def ground_truth_label(video, ground_truth_file, block):
    """
    Ground truth count from 0, inclusive, in format:
    video (w/ extension), 1st fake frame, last fake frame, # number of frames

    return: Nx1 array with 1 is fake, 0 is genuine
    """
    data = pd.read_csv(ground_truth_file, header=0, index_col=0)
    if video not in data.index:
        return [], -1, -1, []
    start, end, total = data.loc[video].tolist()
    labels = np.zeros((total,), dtype=int)  # individual frame labels
    labels[start:end + 1] = 1
    block_labels = []  # labels for a block of frames

    # vectors = np.array(labels)
    for i in range(0, len(labels) - block, block):
        lbl_block = list(labels[i:i + block])
        fake_frames = lbl_block.count(1)  # fake=1
        real_frames = lbl_block.count(0)  # real=0
        curr_block_label = 1 if fake_frames > real_frames else 0
        block_labels.append(curr_block_label)

    return labels, start, end, block_labels


def get_data_for_test(path, file, fake, block):  # fake: fake=1 genuine=0
    files = os.listdir(path)
    x = []
    x_diff = []
    y = []

    video_y = []
    count_y = {}
    sample_to_video = []

    # print("Loading data and embedding...")
    # for file in tqdm(files):
    vectors = np.loadtxt(os.path.join(path, file))
    video_y.append(fake)

    for i in range(0, vectors.shape[0] - block, block):
        vec = vectors[i:i + block, :]
        x.append(vec)
        vec_next = vectors[i + 1:i + block, :]
        vec_next = np.pad(vec_next, ((0, 1), (0, 0)), 'constant', constant_values=(0, 0))
        vec_diff = (vec_next - vec)[:block - 1, :]
        x_diff.append(vec_diff)

        y.append(fake)

        # Dict for counting number of samples in video
        if file not in count_y:
            count_y[file] = 1
        else:
            count_y[file] += 1

        # Recording each samples belonging
        sample_to_video.append(file)
    return np.array(x), np.array(x_diff), np.array(y), np.array(video_y), np.array(sample_to_video), count_y


def merge_video_prediction(mix_prediction, s2v, vc):
    prediction_video = []
    pre_count = {}
    for p, v_label in zip(mix_prediction, s2v):
        p_bi = 0
        if p >= 0.5:
            p_bi = 1
        if v_label in pre_count:
            pre_count[v_label] += p_bi
        else:
            pre_count[v_label] = p_bi
    for key in pre_count.keys():
        prediction_video.append(pre_count[key] / vc[key])
    return prediction_video


def evaluate(ground_truth, predictions):
    """
    Compute the Jaccard similarity = Intersection / Union.
    Return the intersection, union and the similarity score
    """
    assert len(predictions) == len(ground_truth)
    n_frames = len(predictions)
    intersect = sum(1 for x in range(n_frames) if (predictions[x] == ground_truth[x]))
    union = intersect + sum(1 for x in range(n_frames) if (predictions[x] != ground_truth[x])) * 2

    return intersect, union, intersect / union


def main(args):
    video_selection = args.video_selection
    model_selection = args.model_selection

    gt_file_path = r'.\sample_ground_truth.csv'
    assert os.path.exists(gt_file_path), "Ground path does not exist. Please make sure the file exists."

    landmark_path = os.path.join(r'.\landmarks', video_selection)
    assert os.path.exists(landmark_path), "Landmark path does not exist. Please extract the landmarks firstly."

    result_csv_file_path = os.path.join(r'.\results', 'vid-' + video_selection + '_model-' + model_selection + '.csv')
    assert not os.path.exists(result_csv_file_path), "Results file already exists. Experiment might be already done."
    result_csv_file = open(result_csv_file_path, 'w')

    video_filenames = [f[:-4] for f in os.listdir(landmark_path) if f[-4:] == '.txt']
    # video_filenames = video_filenames[:5]  # for debug, delete after TODO
    for video in tqdm(video_filenames):
        gt_labels, start, end, gt_block_labels = ground_truth_label(video, gt_file_path, block_size)
        if not len(gt_labels) or start < 0 or end < 0:  # not found in ground truth file
            continue

        test_samples, test_samples_diff, _, _, test_sv, test_vc = get_data_for_test(landmark_path, video + '.txt', 1,
                                                                                    block_size)
        model = K.Sequential([
            layers.InputLayer(input_shape=(block_size, 136)),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.GRU(RNN_UNIT)),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(64, activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(2, activation='softmax')
        ])
        model_diff = K.Sequential([
            layers.InputLayer(input_shape=(block_size - 1, 136)),
            layers.Dropout(0.25),
            layers.Bidirectional(layers.GRU(RNN_UNIT)),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(64, activation='relu'),
            layers.Dropout(DROPOUT_RATE),
            layers.Dense(2, activation='softmax')
        ])

        lossFunction = K.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = K.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                      loss=lossFunction,
                      metrics=['accuracy'])
        model_diff.compile(optimizer=optimizer,
                           loss=lossFunction,
                           metrics=['accuracy'])

        if model_selection == 'DeeperForensics':
            # ----Using Deeperforensics 1.0 Parameters----#
            model.load_weights('./model_weights/deeper/g1.h5')
            model_diff.load_weights('./model_weights/deeper/g2.h5')
        elif model_selection == 'FFpp':
            # ----Using FF++ Parameters----#
            model.load_weights('./model_weights/ff/g1.h5')
            model_diff.load_weights('./model_weights/ff/g2.h5')

        prediction = model.predict(test_samples)
        prediction_diff = model_diff.predict(test_samples_diff)

        for i in range(len(prediction)):
            mix = prediction[i][1] + prediction_diff[i][1]
            mixed_pred = 0 if mix / 2 < mixed_pred_threshold else 1
            mixed_predictions.append(mixed_pred)
        mixed_predictions = smooth_predictions(mixed_predictions)
        intersection, union, iou = evaluate(gt_block_labels, mixed_predictions) 
        result_csv_file.write(f'{video},{iou:.2f}\n')
        print(f'{video=}, {iou=:.2f}')
    result_csv_file.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract landmarks sequences from input videos.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-v', '--video_selection', type=str, default='Face2Face',
                        help="Either 'Face2Face' or 'NeuralTexture'"
                        )
    parser.add_argument('-m', '--model_selection', type=str, default='FFpp',
                        help="Either 'FFpp' or 'DeeperForensics'"
                        )
    args = parser.parse_args()
    main(args)
