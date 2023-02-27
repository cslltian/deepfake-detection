import numpy as np
import pandas as pd



def grouth_truth_label(video, grouth_truth_file):
    """
    Groundtruth count from 0, inclusive, in format:
    video (w/ extension), 1st fake frame, last fake frame, # number of frames

    return: Nx1 array with 1 is fake, 0 is genuine
    """
    data = pd.read_csv(grouth_truth_file, header=None, index_col=0)
    start, end, total = data.loc[video].tolist()
    # print(start, end, total)

    labels = np.zeros((total,), dtype=int)
    labels[start:end+1] = 1
    return labels


def jaccard_similarity(predicted, ground_truth):
    """
    Compute the Jaccard similarity = Intersection / Union.
    Return the intersection, union and the similarity score
    """
    n_frames = len(predicted)
    intersect = sum(1 for x in range(n_frames) if (predicted[x] == 1 and ground_truth[x] == 1))
    union = sum(1 for x in range(n_frames) if (predicted[x] == 1 or ground_truth[x] == 1))

    return intersect, union, intersect/union


def compute_false_accepted(predicted, ground_truth):
    # False accept == ground_truth is real & predicted is fake
    # True accept == ground_truth is fake & predicted is fake
    n_frames = len(predicted)
    n_false_accept = sum(1 for x in range(n_frames) if (ground_truth[x] == 0 and predicted[x] == 1))
    n_true_accept = sum(1 for x in range(n_frames) if (ground_truth[x] == 1 and predicted[x] == 1))
    FAR = n_false_accept / (n_false_accept + n_true_accept)
    return n_false_accept, FAR


def compute_false_rejected(predicted, ground_truth):
    # False reject == ground_truth is fake & predicted is real
    # True reject == ground_truth is real & predicted is real
    n_frames = len(predicted)
    n_false_reject = sum(1 for x in range(n_frames) if (ground_truth[x] == 1 and predicted[x] == 0))
    n_true_reject = sum((1 for x in range(n_frames) if (ground_truth[x] == 0 and predicted[x] == 0)))
    FRR = n_false_reject / (n_false_reject + n_true_reject)
    return n_false_reject, FRR


def compute_precision(predicted, ground_truth):
    n_correct = sum(np.array(predicted) == np.array(ground_truth))

    return n_correct / len(predicted)