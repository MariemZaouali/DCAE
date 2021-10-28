#####
"""
If you plan on using this implementation, please cite our work (https://ieeexplore.ieee.org/abstract/document/8948005):
@ARTICLE{Nalepa2020_3DCAE,
 author={J. {Nalepa} and M. {Myller} and Y. {Imai} and K. -I. {Honda} and T. {Takeda} and M. {Antoniak}},
 journal={IEEE Geoscience and Remote Sensing Letters},
 title={Unsupervised Segmentation of Hyperspectral Images Using 3-D Convolutional Autoencoders},
 year={2020},
 volume={17},
 number={11},
 pages={1948-1952},
 doi={10.1109/LGRS.2019.2960945}}
"""
import os
#import utils
#import io
#import data_structures
from typing import Tuple

from typing import Iterable
import numpy as np
from scipy.io import loadmat
import numpy as np
from time import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from skimage.io import imsave
from skimage.color import label2rgb
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
#find another clustering method without number of cluster setup
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

#utils
# -*- coding: utf-8 -*-
import hdf5storage
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
from scipy import io, misc
import imageio
import os
import re
import torch
import abc
from copy import copy
from math import ceil
from os import PathLike
from random import shuffle
from itertools import product
from collections import Iterable

import torch
import numpy as np
from typing import List
from tensorflow.keras.utils import to_categorical




def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.
    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)
    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.
    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)
    Returns:
        arr_2d: int 2D array of labels
    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption=""):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption})
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption})

def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.
    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any
    """
    print("Image has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.
    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class
    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.
    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.
    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples
    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image
    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window
    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window
    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size
    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.
    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).
    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    #target = target[ignored_mask] -1
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


def show_results(results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        kappa = results["Kappa"]

    #label_values = label_values[1:]
    vis.heatmap(cm, opts={'title': "Confusion matrix", 
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.03f} +- {:.03f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.03f}%\n".format(accuracy)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("Kappa: {:.03f} +- {:.03f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.03f}\n".format(kappa)

    vis.text(text.replace('\n', '<br/>'))
    print(text)


def sample_gt(gt, train_size, mode='random'):
    """Extract a fixed percentage of samples from an array of labels.
    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.
    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.
    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.
    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients 
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()
#io
def load_data(path: os.PathLike):
    """
    Loading data from NumPy array format (.npy) or from MATLAB format (.mat)
    :param path: Path to either .npy or .mat type file
    :return: numpy array with loaded data
    """
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = hdf5storage.loadmat(path)#io.loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    else:
        raise ValueError("This file type is not supported")
    print("data returned",data.shape)
    return data


def save_to_csv(path: os.PathLike, to_save: Iterable, mode: str='a'):
    """
    Save an iterable to a CSV file
    :param path: Path to the file
    :param to_save: An iterable containing data to be saved
    :param mode: Mode in which the file will be opened
    :return: None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _, extension = os.path.splitext(path)
    if extension != ".csv" and extension != '':
        path.replace(extension, ".csv")
    elif extension == '':
        path += ".csv"
    csv = open(path, mode=mode)
    to_save_string = ",".join(str(x) for x in to_save) + "\n"
    csv.write(to_save_string)
    csv.close()

#data_structures

HEIGHT = 0
WIDTH = 1
DEPTH = 2


class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def get_data(self) -> np.ndarray:
        """
        :return: Data from a given dataset
        """
        return self.data

    def get_labels(self) -> np.ndarray:
        """
        :return: Labels from a given dataset
        """
        return self.labels

    def get_one_hot_labels(self, classes_count: int=None):
        if classes_count is None:
            classes_count = len(np.unique(self.labels))
        return to_categorical(self.labels, classes_count)

    @property
    def min(self):
        return np.amin(self.data)

    @property
    def max(self):
        return np.amax(self.data)

    @property
    def shape(self):
        return self.data.shape

    def vstack(self, to_stack: np.ndarray):
        self.data = np.vstack([self.data, to_stack])

    def hstack(self, to_stack: np.ndarray):
        self.labels = np.hstack([self.labels, to_stack])

    def expand_dims(self, axis: int=0, inplace: bool=True):
        if inplace:
            self.data = np.expand_dims(self.data, axis=axis)
        else:
            return np.expand_dims(self.data, axis=axis)

    def normalize_min_max(self, min_: float=None, max_: float=None,
                          inplace: bool=True):
        """
        Normalize data using Min Max normalization: (data - min) / (max - min)
        :param min_: Minimal value for normalization, if not specified,
                     it will be deducted from data
        :param max_: Maximal value for normalization, if not specified,
                     it will be deducted from data
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return: If inplace is True - return None,
                 if inplace is False - return normalized (data, labels)
        """
        if min_ is None and max_ is None:
            min_ = np.amin(self.get_data())
            max_ = np.amax(self.get_data())
            if inplace:
                self.data = (self.get_data() - min_) / (max_ - min_)
            else:
                return (self.get_data() - min_) / (max_ - min_)
        elif min_ is not None and max_ is not None:
            if inplace:
                self.data = (self.get_data() - min_) / (max_ - min_)
            else:
                return(self.get_data() - min_) / (max_ - min_)

    def standardize(self, mean: float=None, std: float=None,
                    inplace: bool=True):
        """
        Standardize data using mean and std.
        :param mean: Mean value for standardization, if not specified,
                     it will be deducted from data
        :param std: Std value for standardization, if not specified,
                     it will be deducted from data
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return: If inplace is True - return None,
                 if inplace is False - return normalized (data, labels)
        """
        if mean is None and std is None:
            mean = np.mean(self.get_data())
            std = np.std(self.get_data())
        if inplace:
            self.data = (self.data - mean) / std
        else:
            return (self.data - mean) / std

    def normalize_labels(self):
        """
        Normalize label values so that they start from 0.
        :return: None
        """
        self.labels = self.labels - 1

    def delete_by_indices(self, indices: Iterable):
        """
        Delete a chunk of data given as indices
        :param indices: Indices to delete from both data and labels arrays
        :return: None
        """
        self.data = np.delete(self.data, indices, axis=HEIGHT)
        self.labels = np.delete(self.labels, indices, axis=HEIGHT)

    def convert_to_tensors(self, inplace: bool=True, device: str='cpu'):
        """
        Convert data and labels from torch tensors.
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :param device: Device on which tensors should be alocated
        :return:
        """
        if inplace:
            self.data = torch.from_numpy(self.get_data()).float().to(device)
            self.labels = torch.from_numpy(self.get_labels()).float().to(device)
        else:
            return torch.from_numpy(self.get_data()).to(device), \
                   torch.from_numpy(self.get_labels()).to(device)

    def convert_to_numpy(self, inplace: bool=True):
        """
        Convert data and labels to numpy
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return:
        """
        if inplace:
            self.data = self.data.numpy()
            self.labels = self.labels.numpy()
        else:
            return self.data.numpy(), self.labels.numpy()

    def __len__(self) -> int:
        """
        Method providing a size of the dataaset (number of samples)
        :return: Size of the dataset
        """
        return len(self.labels)

    def __getitem__(self, item) -> [np.ndarray, np.ndarray]:
        """
        Method supporting integer indexing
        :param item: Index or Iterable of indices pointing at elements to be
                     returned
        :return: Data at given indexes
        """
        sample_x = self.data[item, ...]
        sample_y = self.labels[item]
        return sample_x, sample_y


class HyperspectralDataset(Dataset):
    """
    Class representing hyperspectral data in a form of samples prepared for
    training and classification (1D or 3D). For 1D samples, data will have
    the following dimensions: [SAMPLES_COUNT, NUMBER_OF_BANDS], where for 3D
    samples dimensions will be [SAMPLES_COUNT,
                                NEIGHBOURHOOD_SIZE,
                                NEIGHBOURHOOD_SIZE,
                                NUMBER_OF_BANDS].
    """
    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike],
                 neighbourhood_size: int = 1,
                 background_label: int = 0):
        if type(dataset) is np.ndarray and type(ground_truth) is np.ndarray:
            raw_data = dataset
            ground_truth = ground_truth
        elif type(dataset) is str and type(ground_truth) is str:
            raw_data = load_data(dataset)
            ground_truth = load_data(ground_truth)
        else:
            raise TypeError("Dataset and ground truth should be "
                            "provided either as a string or a numpy array, "
                            "not {}".format(type(dataset)))
        data, labels = self._prepare_samples(raw_data,
                                             ground_truth,
                                             neighbourhood_size,
                                             background_label)
        super(HyperspectralDataset, self).__init__(data, labels)

    @staticmethod
    def _get_padded_cube(data, padding_size):
        x = copy(data)
        v_padding = np.zeros((padding_size, x.shape[WIDTH], x.shape[DEPTH]))
        x = np.vstack((v_padding, x))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[HEIGHT], padding_size, x.shape[DEPTH]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    @staticmethod
    def _prepare_1d(raw_data: np.ndarray,
                    ground_truth: np.ndarray,
                    background_label: int):
        samples, labels = list(), list()
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(raw_data[y, x, ...])
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def _prepare_3d(self, raw_data: np.ndarray,
                    ground_truth: np.ndarray,
                    neighbourhood_size: int,
                    background_label: int):
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        padding_size = neighbourhood_size % ceil(float(neighbourhood_size) / 2.)
        padded_cube = self._get_padded_cube(raw_data, padding_size)
        samples, labels = list(), list()
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(padded_cube[y:y + padding_size * 2 + 1,
                                          x:x + padding_size * 2 + 1, ...])
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def _prepare_samples(self, raw_data: np.ndarray,
                         ground_truth: np.ndarray,
                         neighbourhood_size: int,
                         background_label: int):
        if neighbourhood_size > 1:
            samples, labels = self._prepare_3d(raw_data,
                                               ground_truth,
                                               neighbourhood_size,
                                               background_label)
        else:
            samples, labels = self._prepare_1d(raw_data,
                                               ground_truth,
                                               background_label)
        return (np.array(samples).astype(np.float64),
                np.array(labels).astype(np.uint8))


class Subset(abc.ABC):

    @abc.abstractmethod
    def extract_subset(self, *args, **kwargs) -> [np.ndarray, np.ndarray]:
        """"
        Extract some part of a given dataset
        """


class BalancedSubset(Dataset, Subset):
    """
    Extracted a subset where all classes have the same number of samples
    """

    def __init__(self, dataset: Dataset,
                 samples_per_class: int,
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset,
                                           samples_per_class,
                                           delete_extracted)
        super(BalancedSubset, self).__init__(data, labels)

    @staticmethod
    def _collect_indices_to_extract(classes: List[int],
                                    labels: np.ndarray,
                                    samples_per_class: int):
        indices_to_extract = []
        for label in classes:
            class_indices = list(np.where(labels == label)[0])
            shuffle(class_indices)
            if 0 < samples_per_class < 1:
                samples_to_extract = int(len(class_indices) * samples_per_class)
                indices_to_extract += class_indices[0:samples_to_extract]
            else:
                indices_to_extract += class_indices[0:int(samples_per_class)]
        return indices_to_extract

    def extract_subset(self, dataset: Dataset,
                       samples_per_class: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        classes, counts = np.unique(dataset.get_labels(), return_counts=True)
        if np.any(counts < samples_per_class):
            raise ValueError("Chosen number of samples per class is too big "
                             "for one of the classes")
        indices_to_extract = self._collect_indices_to_extract(classes,
                                                              dataset.get_labels(),
                                                              samples_per_class)

        data = copy(dataset.get_data()[indices_to_extract, ...])
        labels = copy(dataset.get_labels()[indices_to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(indices_to_extract)

        return data, labels


class ImbalancedSubset(Dataset, Subset):
    """
    Extract a subset where samples are drawn randomly. If total_samples_count
    is a value between 0 and 1, it is treated as a percentage of dataset
    to extract.
    """
    def __init__(self, dataset: Dataset,
                 total_samples_count: float,
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset,
                                           total_samples_count,
                                           delete_extracted)
        super(ImbalancedSubset, self).__init__(data, labels)

    def extract_subset(self, dataset: Dataset,
                       total_samples_count: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        indices = [i for i in range(len(dataset))]
        shuffle(indices)
        if 0 < total_samples_count < 1:
            total_samples_count = int(len(dataset) * total_samples_count)
        indices_to_extract = indices[0:total_samples_count]

        data = copy(dataset.get_data()[indices_to_extract, ...])
        labels = copy(dataset.get_labels()[indices_to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(indices_to_extract)

        return data, labels


class CustomSizeSubset(Dataset, Subset):
    """
    Extract a subset where number of samples for each class is provided
    separately in a list
    """
    def __init__(self, dataset: Dataset,
                 samples_count: List[int],
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset, samples_count,
                                           delete_extracted)
        super(CustomSizeSubset, self).__init__(data, labels)

    def extract_subset(self, dataset: Dataset, samples_count: List[int],
                       delete_extracted: bool):
        classes = np.unique(dataset.get_labels())
        to_extract = []
        for label in classes:
            indices = np.where(dataset.get_labels() == label)[0]
            shuffle(indices)
            to_extract += list(indices[0:samples_count[label]])

        data = copy(dataset.get_data()[to_extract, ...])
        labels = copy(dataset.get_labels()[to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(to_extract)

        return data, labels


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets. Useful when loading patches
    of the dataset and combining them"""

    def __init__(self, datasets: List[Dataset]):
        data, labels = self.combine_datasets(datasets)
        super(ConcatDataset, self).__init__(data, labels)

    @staticmethod
    def combine_datasets(datasets: List[Dataset]) -> [np.ndarray, np.ndarray]:
        data = [dataset.get_data() for dataset in datasets]
        labels = [dataset.get_labels() for dataset in datasets]
        return np.vstack(data), np.hstack(labels)


class OrderedDataLoader:
    """
    Shuffling is performed only withing classes, the order of the
    returned classes is fixed.
    """
    def __init__(self, dataset: Dataset, batch_size: int=64,
                 use_tensors: bool=True):
        self.batch_size = batch_size
        self.label_samples_indices = self._get_label_samples_indices(dataset)
        self.samples_returned = 0
        self.samples_count = len(dataset)
        self.indexes = self._get_indexes()
        self.data = dataset
        if use_tensors:
            self.data.convert_to_tensors()

    def __iter__(self):
        self.indexes = self._get_indexes()
        return self

    def __next__(self):
        if (self.samples_returned + self.batch_size) > self.samples_count:
            self.samples_returned = 0
            raise StopIteration
        else:
            indexes = self.indexes[self.samples_returned:
                                   self.samples_returned + self.batch_size]
            batch = self.data[indexes]
            self.samples_returned += self.batch_size
            return batch

    def _get_indexes(self):
        labels = self.label_samples_indices.keys()
        indexes = []
        for label in labels:
            shuffle(self.label_samples_indices[label])
            indexes += self.label_samples_indices[label]
        return indexes

    @staticmethod
    def _get_label_samples_indices(dataset):
        labels = np.unique(dataset.get_labels())
        label_samples_indices = dict.fromkeys(labels)
        for label in label_samples_indices:
            label_samples_indices[label] = list(np.where(dataset.get_labels() == label)[0])
        return label_samples_indices


class HyperspectralCube(Dataset):
    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike] = None,
                 neighbourhood_size: int = 1,
                 device: str='cpu',
                 bands=None):
        if type(dataset) is np.ndarray and type(ground_truth) is np.ndarray:
            raw_data = dataset
            ground_truth = ground_truth
        elif type(dataset) is str and type(ground_truth) is str:
            raw_data = load_data(dataset)
            ground_truth = load_data(ground_truth)
        elif type(dataset) is str and ground_truth is None:
            raw_data = load_data(dataset)
        else:
            raise TypeError("Dataset and ground truth should be "
                            "provided either as a string or a numpy array, "
                            "not {}".format(type(dataset)))
        self.neighbourhood_size = neighbourhood_size
        self.original_2d_shape = raw_data.shape[0:2]
        self.padding_size = neighbourhood_size % ceil(float(neighbourhood_size) / 2.)
        self.indexes = self._get_indexes(raw_data.shape[HEIGHT], raw_data.shape[WIDTH])
        data = self._get_padded_cube(raw_data)
        data = data.swapaxes(1, 2).swapaxes(0, 1)
        self.device = device
        self.bands = bands
        super(HyperspectralCube, self).__init__(data, ground_truth)

    @staticmethod
    def _get_indexes(height, width):
        xx, yy = np.meshgrid(range(height), range(width))
        return [(x, y) for x, y in zip(yy.flatten(), xx.flatten())]

    def _get_padded_cube(self, data):
        v_padding = np.zeros((self.padding_size, data.shape[WIDTH], data.shape[DEPTH]))
        x = np.vstack((v_padding, data))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[HEIGHT], self.padding_size, x.shape[DEPTH]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        if type(item) is list:
            batch = torch.zeros([len(item),
                                 1,
                                 self.bands,
                                 self.neighbourhood_size,
                                 self.neighbourhood_size], device=self.device)
            for sample, sample_index in enumerate(item):
                x, y = self.indexes[sample_index]
                batch[sample, 0] = self.data[:self.bands, y:y + self.padding_size * 2 + 1,
                                                x:x + self.padding_size * 2 + 1]
            return batch
        else:
            x, y = self.indexes[item]
            return self.data[:self.bands, y:y + self.padding_size * 2 + 1,
                                x:x + self.padding_size * 2 + 1]


class DataLoaderShuffle:
    def __init__(self, dataset: Dataset, batch_size: int=64):
        self.batch_size = batch_size
        self.data = dataset
        self.samples_count = len(dataset)
        self.indexes = self._get_indexes()
        self.samples_returned = 0

    def __iter__(self):
        self.samples_returned = 0
        return self

    def __next__(self):
        if (self.samples_returned + self.batch_size) > self.samples_count:
            raise StopIteration
        else:
            indexes = self.indexes[self.samples_returned:
                                   self.samples_returned + self.batch_size]
            batch = self.data[indexes]
            self.samples_returned += self.batch_size
            return batch

    def __len__(self):
        return int(self.samples_count / self.batch_size)

    def cube_2d_shape(self):
        return self.data.original_2d_shape

    def shuffle(self):
        shuffle(self.indexes)

    def sort(self):
        self.indexes = self._get_indexes()

    def _get_indexes(self):
        indexes = [x for x in range(self.samples_count)]
        return indexes

		
class DCEC(nn.Module):
    def __init__(self, input_dims: np.ndarray, n_clusters: int,
                 kernel_shape: np.ndarray, last_out_channels: int = 32, latent_vector_size: int = 25,
                 update_interval: int = 140, device: str='cpu',
                 artifacts_path: str='DCEC'):
        super(DCEC, self).__init__()
        self.latent_vector_size = latent_vector_size
        self.n_clusters = n_clusters
        self.last_out_channels = last_out_channels
        encoder_shape, out_features = self._calculate_shapes(input_dims, kernel_shape, 2,
                                                             self.last_out_channels)
        self.final_encoder_shape = tuple(np.hstack([self.last_out_channels, encoder_shape]))
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=kernel_shape),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv3d(in_channels=32, out_channels=self.last_out_channels,
                      kernel_size=kernel_shape),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=out_features, out_features=self.latent_vector_size)
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.latent_vector_size, out_features=out_features),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.last_out_channels, out_channels=32,
                               kernel_size=kernel_shape),
            nn.ReLU(),
            nn.Dropout(),
            nn.ConvTranspose3d(in_channels=32, out_channels=1,
                               kernel_size=kernel_shape)
        )
        self.clustering_layer = ClusteringLayer(n_clusters=self.n_clusters,
                                                input_dim=self.latent_vector_size)
        self.log_softmax = nn.LogSoftmax()
        self.update_interval = update_interval
        self.n_clusters = n_clusters
        self.artifacts_path = artifacts_path
        self.mse_loss = nn.MSELoss()
        self.kld_loss = nn.KLDivLoss(reduction='batchmean')
        self.device = device
        self.metrics = {'MSE': [],
                        'KLD': [],
                        'NMI': [0],
                        'ARS': [0]}
        self._best_autoencoder_loss = np.inf
        self._best_nmi = -np.inf

    def _calculate_shapes(self, input_dims: np.ndarray, kernel_shapes: np.ndarray, kernels_count: int,
                          channels_count: int):
        final_encoder_shape = input_dims - ((kernels_count * kernel_shapes) - kernels_count)
        out_features = np.prod(final_encoder_shape) * channels_count
        return final_encoder_shape, out_features

    #I have to change here!!
    def predict_clusters(self, data_loader):
        predicted_clusters = torch.zeros((len(data_loader) *
                                        data_loader.batch_size, self.n_clusters), device=self.device)
        last_insert = 0
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = Variable(batch_x, requires_grad=False).float()
                predicted_clusters[last_insert:
                                   last_insert +
                                   data_loader.batch_size, :] = self.clustering_layer(self.encoder(batch_x))
                last_insert += data_loader.batch_size
        return predicted_clusters

    @staticmethod
    def calculate_target_distribution(q):
        weight = torch.pow(q, 2) / torch.sum(q, dim=0)
        return torch.t(torch.t(weight) / torch.sum(weight, dim=1))

    def get_target_distribution(self, data_loader: DataLoader):
        return self.calculate_target_distribution(
            self.predict_clusters(data_loader))

    def encode_features(self, data_loader: DataLoader):
        encoded_features = torch.zeros((len(data_loader) *
                                        data_loader.batch_size, self.latent_vector_size), device=self.device)
        last_insert = 0
        with torch.no_grad():
            for batch_x in data_loader:
                batch_x = Variable(batch_x, requires_grad=False).float()
                encoded_features[last_insert:
                                 last_insert +
                                 data_loader.batch_size, :] = self.encoder(batch_x)
                last_insert += data_loader.batch_size
        return encoded_features

    #contains cluster initialization
    #I have to change the weight here!!
    def initialize(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        print("type of encoded", type(encoded_features.cpu().detach()))
        # The following bandwidth can be automatically detected using
        """
        bandwidth = estimate_bandwidth(encoded_features.cpu().detach(), quantile=0.2,n_samples=1000)#(len(data_loader)*data_loader.batch_size* self.latent_vector_size)//2)
        ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
        ms.fit(encoded_features.cpu().detach())
        labels = ms.labels_
        #cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters = len(labels_unique)
        self.n_clusters=n_clusters
        """
        kmeans = KMeans(n_clusters=self.n_clusters).fit(encoded_features.cpu().detach())
        cluster_centers = kmeans.cluster_centers_.astype(np.float32)
        cluster_centers = torch.from_numpy(cluster_centers).to(device=self.device)
        self.clustering_layer.set_weights(cluster_centers)
  
        print("the number of cluster",self.n_clusters)
        
    def train_with_clustering(self, data_loader, optimizer, iterations: int,
                              gamma: float):
        self.initialize(data_loader)
       
        last_batch = 0
        true_labels = (data_loader.data.labels
                       .cpu()
                       .detach()
                       .numpy()
                       .transpose()
                       .reshape(-1))
        for iteration in range(iterations):
            if iteration % self.update_interval == 0:
                last_batch = 0
                data_loader.sort()
                predicted_labels = self.cluster_with_model(data_loader)
                self.plot_high_res(predicted_labels, data_loader.cube_2d_shape(),
                          iteration, 'model')
                self.metrics['NMI'].append(self.calculate_nmi(true_labels,
                                                              predicted_labels))
                self.metrics['ARS'].append(self.calculate_ars(true_labels,
                                                              predicted_labels))
                self._log_metrics_to_file()
                self._print_losses(iteration)
                self._save_model(iteration)
                data_loader.shuffle()
                target_distribution = self.get_target_distribution(data_loader)
                iter(data_loader)
            optimizer.zero_grad()
            try:
                batch_x = next(data_loader)
                batch_x = Variable(batch_x).float()
            except StopIteration:
                iter(data_loader)
                last_batch = 0
                continue
            encoder_output = self.encoder(batch_x)
            #here where the truly clustering result is got
            #I have to change in clustering layer class!
            clustering_layer_output = self.log_softmax(self.clustering_layer(encoder_output))
            div_loss = self.kld_loss(clustering_layer_output,
                                     target_distribution[last_batch:
                                                         last_batch +
                                                         data_loader.batch_size]) * gamma
            linear_output = self.linear(encoder_output)
            linear_output = torch.reshape(linear_output,
                                          ((data_loader.batch_size, ) + self.final_encoder_shape))
            decoder_output = self.decoder(linear_output)
            mse_loss = self.mse_loss(batch_x, decoder_output)
            self.metrics['MSE'].append(mse_loss.item())
            self.metrics['KLD'].append(div_loss.item())
            div_loss.backward(retain_graph=True)
            mse_loss.backward()
            optimizer.step()
            last_batch += data_loader.batch_size

    def train_autoencoder(self, data_loader, optimizer, epochs, epsilon):
        true_labels = (data_loader.data.labels
                       .cpu()
                       .detach()
                       .numpy()
                       .transpose()
                       .reshape(-1))
        #added by myself
        self.initialize(data_loader)
        last_mse = 0
        for epoch in range(epochs):
            data_loader.shuffle()
            for batch_x in data_loader:
                batch_x = Variable(batch_x).float()
                optimizer.zero_grad()
                encoded = self.encoder(batch_x)
                linear_output = self.linear(encoded)
                reshaped = torch.reshape(linear_output,
                                         ((data_loader.batch_size, ) + self.final_encoder_shape))
                decoder_output = self.decoder(reshaped)
                mse_loss = self.mse_loss(batch_x, decoder_output)
                self.metrics['MSE'].append(mse_loss.item())
                mse_loss.backward()
                optimizer.step()
            data_loader.sort()
            self.save_if_best(np.average(self.metrics['MSE']))
            predicted_labels = self.cluster_with_kmeans(data_loader)
            #predicted_labels = self.cluster_with_gaussian(data_loader)
            #predicted_labels = self.cluster_with_dbscan(data_loader)
            #predicted_labels = self.cluster_with_meanshift(data_loader)
            self.plot_high_res(predicted_labels, data_loader.cube_2d_shape(),
                      epoch, 'kmeans')
            self.metrics['NMI'].append(self.calculate_nmi(true_labels,
                                                          predicted_labels))
            self.metrics['ARS'].append(self.calculate_ars(true_labels,
                                                          predicted_labels))
            self._log_metrics_to_file()
            if epoch > 1:
                if np.abs(last_mse - np.average(self.metrics['MSE'])) < epsilon:
                    break
            last_mse = np.average(self.metrics['MSE'])
            self._print_losses(epoch)

    @staticmethod
    def calculate_nmi(labels_true, labels_predicted):
        to_delete = np.where(labels_true == 0)[0]
        labels_predicted = np.delete(labels_predicted, to_delete)
        labels_true = np.delete(labels_true, to_delete).astype(np.int32)
        return normalized_mutual_info_score(labels_true, labels_predicted)

    @staticmethod
    def calculate_ars(labels_true, labels_predicted):
        to_delete = np.where(labels_true == 0)[0]
        labels_predicted = np.delete(labels_predicted, to_delete)
        labels_true = np.delete(labels_true, to_delete)
        return adjusted_rand_score(labels_true, labels_predicted)

    def _print_losses(self, iteration):
        print('Iter: {}, MSE -> {} KLD -> {}, NMI -> {}, ARS -> {}'
              .format(iteration,
                      np.average(self.metrics['MSE']),
                      np.average(self.metrics['KLD'])
                                    if len(self.metrics['KLD']) != 0 else 0,
                      self.metrics['NMI'][-1],
                      self.metrics['ARS'][-1]))
        self.metrics['KLD'] = []
        self.metrics['MSE'] = []

    def _log_metrics_to_file(self):
        path = os.path.join(self.artifacts_path, 'metrics.csv')
        save_to_csv(path, [np.average(self.metrics['MSE']),
                           np.average(self.metrics['KLD']),
                           self.metrics['NMI'][-1],
                           self.metrics['ARS'][-1]])

    def save_if_best(self, loss):
        if loss < self._best_autoencoder_loss:
            self._save_model()
            self._best_autoencoder_loss = loss

    def train_model(self, data_loader, optimizer, epochs: int=200,
                    iterations: int=10000, gamma: float=0.1, epsilon=0.00001):
        
        print("Pretraining autoencoder:")
        training_start = time()
        self.train_autoencoder(data_loader, optimizer, epochs, epsilon)
        print("Pretraining finished, training with clustering")
        self.load_state_dict(torch.load(os.path.join(self.artifacts_path,
                                                     "best_autoencoder_model.pt")))
        self.train_with_clustering(data_loader, optimizer, iterations, gamma)
        training_time = time() - training_start
        save_to_csv(os.path.join(self.artifacts_path, "time.csv"), [training_time])
        print("Done!")
        
    def cluster_with_dbscan(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        return DBSCAN(eps=3, min_samples=2).fit_predict(
            encoded_features.cpu().detach())

    def cluster_with_meanshift(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        bandwidth = estimate_bandwidth(encoded_features.cpu().detach(), n_samples=500)
        #bandwidth = estimate_bandwidth(encoded_features.cpu().detach(), n_samples=500)#(len(data_loader)*data_loader.batch_size* self.latent_vector_size)//2)
        return MeanShift(bandwidth=bandwidth,bin_seeding=True).fit_predict(
            encoded_features.cpu().detach())
    
    def cluster_with_kmeans(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        return KMeans(n_clusters=self.n_clusters).fit_predict(
            encoded_features.cpu().detach())

    def cluster_with_gaussian(self, data_loader):
        encoded_features = self.encode_features(data_loader)
        return GaussianMixture(n_components=self.n_clusters).fit_predict(
            encoded_features.cpu().detach())

    def cluster_with_model(self, data_loader):
        clusters = self.predict_clusters(data_loader)
        return np.argmax(clusters.cpu().detach().numpy(), axis=1)

    def plot_high_res(self, predicted_labels, shape_2d: Tuple[int, int], epoch: int,
                      clustering_method_name: str, true_labels=None):
        # if true_labels is not None:
        #     predicted_labels[true_labels == 0] = 11
        labels = predicted_labels.reshape(np.flip(shape_2d, axis=0))
        labels = labels.transpose()
        labels = label2rgb(labels)
        dir = os.path.join(self.artifacts_path, clustering_method_name
                           + '_clustering')
        os.makedirs(dir, exist_ok=True)
        file_name = os.path.join(dir, "plot_{}.png".format(epoch))
        imsave(file_name, labels)

    def _save_model(self, epoch: int = None):
        os.makedirs(self.artifacts_path, exist_ok=True)
        if epoch is not None:
            path = os.path.join(self.artifacts_path, "model_epoch_{}.pt".format(epoch))
        else:
            path = os.path.join(self.artifacts_path, "best_autoencoder_model.pt")
        torch.save(self.state_dict(), path)

    def forward(self, *input):
        pass


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters: int, input_dim: int):
        super(ClusteringLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(n_clusters, input_dim))
        nn.init.xavier_normal_(self.weights)

    def set_weights(self, weights):
        self.weights = nn.Parameter(weights).float()

    def forward(self, input_data):
        q = 1.0 / (1.0 + (torch.sum(torch.pow(input_data.unsqueeze(dim=1) -
                                              self.weights, 2), dim=2)))
        q = torch.t(torch.t(q) / torch.sum(q, dim=1))
        return q



    

if __name__ == '__main__':
	print('Change torch.device(cpu) to cuda')
	device ='cuda:0'
	out_path_L = [r'./IP_results',r'./IPf_results']#,r'./PU_results',r'./PUf_results']#r"./"
	data_set_img=[r"./Dataset/IP/indian_all_fmm.npy",r"./Dataset/IP/test_fisher_indian90.npy"]#,r"./Dataset/PaviaU/paviaU_all_fmm.npy",r"./Dataset/PaviaU/test_fisher_paviaU50.npy"]
	data_set_gt=[r"./Dataset/IP/Indian_pines_gt.npy"]#,r"./Dataset/PaviaU/PaviaU_gt.npy"]
			# Example for the Houston dataset
	dataset_bands_L = [392,180]#,392,100]#70
	neighborhood_size = 5
	epochs = 25

	dataset_height_L =[145,145]#,610,610]#610# 1202
	dataset_width_L = [145,145]#,340,340]#340#4768


	batch_size = 25 # The batch size has to be picked in such a way that samples_count % batch_size == 0
	n_clusters_L=[16,9]


	l=[0,0]#,1,1]
	l2=range(0,2)
	for i_gt,i in zip(l,l2):
		out_path=out_path_L[i]
		dataset_bands = dataset_bands_L[i]
		dataset_height=dataset_height_L[i]
		dataset_width=dataset_width_L[i]
		samples_count = dataset_height * dataset_width 
		update_interval = int(samples_count / batch_size)
		iterations = int(update_interval * epochs) # This indicates the number of epochs that the clustering part of the autoencoder will be trained for

			
		dataset = HyperspectralCube(data_set_img[i],data_set_gt[i_gt], # Path to .npy file or np.ndarray with [HEIGHT, WIDTH, BANDS] dimensions
											neighbourhood_size=neighborhood_size,
											device=device, bands=dataset_bands)

				
		dataset.standardize()
		dataset.convert_to_tensors(device=device)
				 # Train
		net = DCEC(input_dims=np.array([dataset_bands, neighborhood_size, neighborhood_size]), n_clusters=n_clusters_L[i_gt],
						   kernel_shape=np.array([5, 3, 3]), latent_vector_size=20,
						   update_interval=update_interval, device=device,
						   artifacts_path=out_path)
		net = net.cuda(device=device)
		optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
		data_loader = DataLoaderShuffle(dataset, batch_size=batch_size)
		net.train_model(data_loader, optimizer, epochs=epochs, iterations=iterations, gamma=0.1)

				# Predict
		net.load_state_dict(torch.load(out_path + "/model_path.pt"))
		predicted_labels = net.cluster_with_model(data_loader)
		net.plot_high_res(predicted_labels, dataset.original_2d_shape, -1, "model")
