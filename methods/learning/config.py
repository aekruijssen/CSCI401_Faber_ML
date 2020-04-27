import argparse

from util import str2bool, str2list


def parse_args():
    parser = argparse.ArgumentParser(
        'yelp_score_predictor',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # dataset
    parser.add_argument('--dataset', type=str, default='yelp_balanced',
                        help='Name of dataset.')

    # model
    parser.add_argument('--model', type=str, default='lstm',
                        help='Name of the model.')
    
    # hardware
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPUs to use for the experiment.')

    # experiment
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for the job.')
    parser.add_argument('--max_global_step', type=int, default=int(1e6),
                        help='Maximum global step to reach.')
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Whether job is used for training.')
    
    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of instances in a batch.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--h_dim', type=int, default=32,
                        help='Number of dimensions in the hidden layers.')
    parser.add_argument('--n_hid', type=int, default=1,
                        help='Number of hidden layers in each module.')
    parser.add_argument('--embedding_dim', type=int, default=50,
                        help='Number of dimensions of word embeddings.')

    # logging
    parser.add_argument('--log_root_dir', type=str, default='logs',
                        help='Path to log root directory.')
    parser.add_argument('--prefix', type=str, default='default',
                        help='Prefix of the run.')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes for the run.')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Training info logging interval.')
    parser.add_argument('--log_img_interval', type=int, default=100,
                        help='Interval to log images.')
    parser.add_argument('--val_interval', type=int, default=10,
                        help='Validation info logging interval.')
    parser.add_argument('--ckpt_interval', type=int, default=100,
                        help='Checkpoint info logging interval.')

    args, unparsed = parser.parse_known_args()
    return args, unparsed
