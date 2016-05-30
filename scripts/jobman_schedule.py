import os
import sys
import argparse

from jobman import sql
from jobman.tools import DD, flatten

from adios.utils.jobman import gen_configurations
from adios.utils.jobman import train_adios
from adios.utils.jobman import train_mlp

def make_argument_parser():
    parser = argparse.ArgumentParser(
        description="Schedule a set of jobman experiments.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--configurations', '-c', required=True,
                        help='The name of YAML file with the hyper '
                             'parameter configurations.')
    parser.add_argument('--dataset', '-ds', required=True,
                        help='The name of the dataset to train models on.')
    parser.add_argument('--labels-order', '-lo', default='labels_order',
                        help='Labels order to use.')
    parser.add_argument('--n-label-splits', '-nl', default=3, type=int,
                        help='Number of ways the labels are split into Y0 and '
                             'Y1 sets. If 0, labels are not split at all; '
                             '0 option should be used only for MLP.')
    parser.add_argument('--batch-size', '-bs', default=128, type=int,
                        help='Training batch size.')
    parser.add_argument('--n-epochs', '-ne', default=100, type=int,
                        help='Maximum number of training epochs to run.')
    parser.add_argument('--model', '-m', required=True,
                        choices=['MLP', 'ADIOS'],
                        help='Name of the model to schedule for training.')
    parser.add_argument('--database', '-db',
                        default='postgres://cisse:cisse2011@biggpu/cisse',
                        help='The address of the database for storing '
                             'the results of the experiments.')
    parser.add_argument('--table-name', '-tn', required=True,
                        help='The name of the table in the database to '
                             'the results of the experiments in.')
    return parser

def main(args):
    # Create a new database
    db = sql.db('%s?table=%s' % (args.database, args.table_name))

    # Create a jobman state
    state = DD()

    # Loop over the search space and schedule jobs
    config_generator = gen_configurations(args.configurations)
    for i, params in enumerate(config_generator):
        # Complete parameters dictionary and add to the state
        state.parameters = params
        state.parameters['model'] = args.model
        state.parameters['dataset'] = args.dataset
        state.parameters['nb_epoch'] = args.n_epochs
        state.parameters['batch_size'] = args.batch_size

        # Insert the job into the database
        if args.model == 'ADIOS':
            state.parameters['labels_order'] = args.labels_order
            state.parameters['n_label_splits'] = args.n_label_splits
            for i in xrange(args.n_label_splits):
                state.parameters['label_split'] = i + 1
                sql.insert_job(train_adios, flatten(state), db)
        else:  # args.model == 'MLP'
            sql.insert_job(train_mlp, flatten(state), db)

    # Create a view for the new database table
    db.createView(args.table_name + '_view')

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    main(args)
