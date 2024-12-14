from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
import torch
import scipy.sparse as sp


class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index])


def load_data(args, test_ratio=0.2):
    """Read data from path, convert data into loader, return features and adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.pos_sample, args.seed))  #
    positive = np.loadtxt(args.pos_sample, dtype=np.int64)   #

    # sample postive
    link_size = int(positive.shape[0])
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    positive = positive[:link_size]
    # print("postive:",positive.shape)

    # sample negative
    negative_all = np.loadtxt(args.neg_sample, dtype=np.int64)
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:positive.shape[0]])



    test_size = int(test_ratio * positive.shape[0]) #测试集
    # print("positive_data:",test_size)

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)  # 96793,3
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1) #96793,3

    # train/test data constrction
    train_data = np.vstack((positive[: -test_size], negative[: -test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))


    # construct adjacency
    train_positive = positive[: -test_size]  # Setting the edges in the test set to 0

    r_d = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),
                        shape=(10367, 1329), dtype=np.float32)
    r_d = r_d.toarray()

    adj = construct_graph(r_d)  #11696, 11696
    adj = lalacians_norm(adj)

    # construct edges
    edges_o = adj.nonzero()
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)

    # extract features
    print('Extracting features...')
    if args.feature_type == 'one_hot':
        features = np.eye(adj.shape[0])

    elif args.feature_type == 'uniform':
        np.random.seed(args.seed)
        features = np.random.uniform(low=0, high=1, size=(adj.shape[0], args.dimensions))

    elif args.feature_type == 'normal':
        np.random.seed(args.seed)
        features = np.random.normal(loc=0, scale=1, size=(adj.shape[0], args.dimensions))

    elif args.feature_type == 'position':
        features = sp.coo_matrix(adj).todense()

    features_o = normalize(features)

    # adversarial nodes
    np.random.seed(args.seed)
    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]

    #original graph
    y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)
    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x = x_o, edge_index=edge_index_o)

    #  shuffle graph
    x_a = torch.tensor(features_a, dtype=torch.float)
    data_a = Data(x=x_a, y=y_a)
    print(data_a)

    print('Loading finished!')
    return data_o, data_a, train_loader, test_loader

