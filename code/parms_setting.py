import argparse


def settings():
    parser = argparse.ArgumentParser()

    # public parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--pos_sample', default="data/RDA.edgelist",
                        help='Path to positive sample.')

    parser.add_argument('--neg_sample', default="data/non_RDA.edgelist",
                        help='Path to negative sample ')

    parser.add_argument('--feature_type', type=str, default='normal', choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='Initial node feature type. Default is normal.')

    parser.add_argument('--model_type', default="CLGNN_GCN", choices=['CLGNN_GCN', 'CLGNN_GIN'],
                        help='. Initial model  type. Default is CLGNN_GIN.')

    # Training settings
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5.')

    parser.add_argument('--weight_decay', default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size. Default is 25.')

    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train. Default is 50.')

    parser.add_argument('--loss_ratio1', type=float, default=1,
                        help='Ratio of task1. Default is 1')

    parser.add_argument('--loss_ratio2', type=float, default=0.5,
                        help='Ratio of task2. Default is 0.1')

    parser.add_argument('--eps', type=float, default=0,
                        help='Initial epsilon value for GIN')

    # model parameter setting
    parser.add_argument('--dimensions', type=int, default=256,
                        help='dimensions of feature d. Default is 256')

    parser.add_argument('--hidden1', default=128,
                        help='Embedding dimension of encoder layer 1 for CLGNN. Default is d/2.')

    parser.add_argument('--hidden2', default=64,
                        help='Embedding dimension of encoder layer 2 for CLGNN. Default is d/4.')

    parser.add_argument('--decoder1', default=256,
                        help='NEmbedding dimension of decoder layer 1 for CLGNN. Default is 256.')

    args = parser.parse_args()

    return args

