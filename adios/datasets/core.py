"""
Generic functionality for multi-label classification datasets.
"""
import os
import sys
import gzip
import logging
import cPickle as pkl

import numpy as np


class MLCDataset(object):
    """
    An abstract class for MLC datasets.
    Create a subclass of this class for your dataset and implement
    __init__ and get_design_matrices functions.

    It provides the essential MLC-specific initial data preprocessing
    functionality through a set of methods.
    """
    def  __init__(self, datadir, which_set, only_labels=False,
                  n_labels=None, take_labels_from_start=True,
                  labels_order='original', min_labels_per_example=1,
                  start=0.0, stop=100.0, dataseed=42, labelseed=42,
                  verbose=False):
        self.datadir = datadir
        self.verbose = verbose

        # Load the dataset
        dataset = self.load_dataset(which_set)
        X, y = self.get_design_matrices(dataset)

        # Sanity checks
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == self.n_features
        self.n_examples = X.shape[0]

        # Apply MLC-specific data preprocessing
        X, y = self.remove_loosely_labeled(X, y, min_labels_per_example)
        assert X.shape[0] == y.shape[0] <= self.n_examples
        X, y = self.shuffle_and_slice(X, y, start, stop, dataseed)
        y = self.reorder_labels(y, labels_order, labelseed)

        # Slice the label space
        if only_labels:
            assert n_labels < y.shape[1], \
                "Something is wrong. " \
                "A part of labels should be used as the input and a part " \
                "as the output."

            self.n_features = n_labels
            self.n_labels = y.shape[1] - self.n_features

            X = y[:,:self.n_features] if take_labels_from_start \
                else y[:,-self.n_features:]
            y = y[:,-self.n_labels:] if take_labels_from_start \
                else y[:,:self.n_labels]

        else:
            y = y[:,:n_labels] if take_labels_from_start else y[:,-n_labels:]

        self.n_examples = X.shape[0]
        self.n_labels = y.shape[1]

        self.X = X
        self.y = y

    def load_dataset(self, which_set):
        if self.verbose:
            sys.stdout.write("Reading data...")

        if which_set not in {'train', 'test', 'full'}:
            raise ValueError(
                'Unrecognized `which_set` value "%s". ' % (which_set,) +
                'Valid values are ["train", "test", "full"].')

        datapath = os.path.join(self.datadir, which_set + '.pkl.gz')
        dataset = pkl.load(gzip.open(datapath))

        if self.verbose:
            sys.stdout.write("Done.\n")

        return dataset

    def get_design_matrices(self, dataset):
        """
        Take the dataset and extract `X` and `y` design matrices.
        Should be implemented by each specific dataset.
        """
        raise NotImplementedError(
            str(type(self)) + " does not implement get_design_matrices.")

    def remove_loosely_labeled(self, X, y, min_labels_per_example):
        preserve_idx = np.sum(y, axis=1) >= min_labels_per_example

        return X[preserve_idx], y[preserve_idx]

    def shuffle_and_slice(self, X, y, start, stop, seed):
        n_examples = X.shape[0]

        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_examples)

        start = int(n_examples * (start / 100.0))
        stop = int(n_examples * (stop / 100.0))
        idx = perm[start:stop]

        return X[idx], y[idx]

    def reorder_labels(self, y, labels_order, seed):
        if labels_order == 'original':
            return y

        elif labels_order == 'random':
            n_labels = y.shape[1]
            rng = np.random.RandomState(seed)
            label_idx = rng.permutation(n_labels).tolist()
        else:
            path = os.path.join(self.datadir, labels_order + '.pkl.gz')
            with gzip.open(path) as fp:
                label_idx = pkl.load(fp)

        return y[:,label_idx]

    def label_cardinality(self):

        return self.y.sum(axis=1).mean()

    def batch_generator(self, batch_size, outputs={}, seed=None):
        rng = np.random.RandomState(seed)

        # Sanity check
        assert batch_size <= self.n_examples and batch_size > 0, \
            "`batch_size` (%d) cannot be greater than the dataset size (%d)." \
            % (batch_size, self.n_examples)

        while True:
            idx = np.asarray(rng.permutation(self.n_examples))
            for i in xrange(0, self.n_examples, batch_size):
                batch_idx = idx[i:min(i + batch_size, self.n_examples)]
                batch = {'X': self.X[batch_idx]}
                for k, v in outputs.iteritems():
                    batch[k] = self.y[batch_idx][:,v]
                yield batch


def balance_labels(datasets):
    """Match labels of the provided datasets.

    Removes columns from the label matrices of each of the datasets that have
    no nonzero entries in at least one of the datasets.
    """
    # Find common labels
    labels = [d.y.astype('bool') for d in datasets]
    label_masks = [y.any(axis=0) for y in labels]
    common_labels = np.all(label_masks, axis=0)
    n_common_labels = common_labels.sum()

    # Keep only the labels that are common between the datasets
    for d in datasets:
        d.y = d.y[:, common_labels]
        d.n_labels = n_common_labels

    return datasets
