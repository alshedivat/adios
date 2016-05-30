import os
import yaml
import numpy as np

from keras.optimizers import Adagrad
from keras.callbacks import EarlyStopping, ModelCheckpoint

from adios.datasets import Delicious as Dataset
from adios.callbacks import HammingLoss
from adios.metrics import f1_measure, hamming_loss, precision_at_k
from adios.utils.assemble import assemble

def main():
    # Load the datasets
    labels_order = 'original'
    train = Dataset(which_set='train', stop=80.0,
                    labels_order=labels_order)
    valid = Dataset(which_set='train', start=80.0,
                    labels_order=labels_order)
    test  = Dataset(which_set='test',
                    labels_order=labels_order)

    nb_features = train.X.shape[1]
    nb_labels = train.y.shape[1]

    # Specify datasets in the format of dictionaries
    train_dataset = {'X': train.X, 'Y': train.y}
    valid_dataset = {'X': valid.X, 'Y': valid.y}
    test_dataset = {'X': test.X, 'Y': test.y}

    # Specify the model
    params = """
    H:
        dim:        1024
        dropout:    0.5
        # batch_norm: {mode: 1}
    """
    params = yaml.load(params)
    params['X'] = {'dim': nb_features}
    params['Y'] = {'dim': nb_labels}

    # Assemble and compile the model
    model = assemble('MLP', params)
    model.compile(loss={'Y': 'binary_crossentropy'},
                  optimizer=Adagrad(1e-1))

    # Make sure checkpoints folder exists
    if not os.path.isdir('checkpoints/'):
        os.makedirs('checkpoints/')

    # Setup callbacks
    callbacks = [
        HammingLoss({'valid': valid_dataset}),
        ModelCheckpoint('checkpoints/mlp_best.h5', monitor='val_hl',
                        verbose=0, save_best_only=True, mode='min'),
        EarlyStopping(monitor='val_hl', patience=15, verbose=0, mode='min'),
    ]

    # Fit the model to the data
    batch_size = 128
    nb_epoch = 300

    model.fit(train_dataset, validation_data=valid_dataset,
              batch_size=batch_size, nb_epoch=nb_epoch,
              callbacks=callbacks, verbose=2)

    # Load the best weights
    if os.path.isfile('checkpoints/mlp_best.h5'):
        model.load_weights('checkpoints/mlp_best.h5')

    # Fit thresholds
    model.fit_thresholds(train_dataset, validation_data=valid_dataset,
                         alpha=np.logspace(-3, 3, num=10).tolist(), verbose=1)

    # Test the model
    probs, preds = model.predict_threshold(test_dataset)

    hl = hamming_loss(test_dataset, preds)
    f1_macro = f1_measure(test_dataset, preds, average='macro')
    f1_micro = f1_measure(test_dataset, preds, average='micro')
    f1_samples = f1_measure(test_dataset, preds, average='samples')
    p_at_1 = precision_at_k(test_dataset, probs, K=1)
    p_at_5 = precision_at_k(test_dataset, probs, K=5)
    p_at_10 = precision_at_k(test_dataset, probs, K=10)

    for k in ['Y']:
        print
        print("Hamming loss (%s): %.2f" % (k, hl[k]))
        print("F1 macro (%s): %.4f" % (k, f1_macro[k]))
        print("F1 micro (%s): %.4f" % (k, f1_micro[k]))
        print("F1 sample (%s): %.4f" % (k, f1_samples[k]))
        print("P@1 (%s): %.4f" % (k, p_at_1[k]))
        print("P@5 (%s): %.4f" % (k, p_at_5[k]))
        print("P@10 (%s): %.4f" % (k, p_at_10[k]))

if __name__ == '__main__':
    main()
