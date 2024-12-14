from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import torch


def normalize(mx):  #
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


'''Construct CDKG graph''' 
def construct_graph(drug_dis):
    drug_shape = drug_dis.shape[0]
    dis_shape = drug_dis.shape[1]

    drug_dis_sim = np.hstack((np.zeros((drug_shape,drug_shape)), drug_dis))
    dis_drug_sim = np.hstack((drug_dis.T, np.zeros((dis_shape,dis_shape))))

    matrix_A = np.vstack((drug_dis_sim, dis_drug_sim))
    return matrix_A



'''Normation'''
def lalacians_norm(adj):
    # adj += np.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)
    return norm_A
