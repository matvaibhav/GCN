import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
nss = 100

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def concatAdjM(adjM, num_simulations):
    adj=[]
    np.zeros(nss *num_simulations)
    for i in range(0, nss *num_simulations):#total simulations become double one for 0 label, one for 1 label
        adj.append(np.zeros(nss *num_simulations))
        # for j in range(0, nss*num_simulations):
        #     adj[i].append(0)
    for i in range(0, num_simulations):

        startIndex=i*nss#nss=1000
        row=0
        col=0
        for j in range(startIndex, startIndex+nss):
            col=0
            for k in range(startIndex, startIndex + nss):
                adj[j][k]=adjM[row][col]
                col += 1
            row += 1
    return adj


def load_data1(dataset_str):
    # """Load data."""
    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))
    #
    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # # from scipy.sparse import *
    # # >> > from scipy import *
    # x = sp.csr_matrix((2, 2), dtype=int)
    #
    # x = [[0, 1], [1, 0]]
    # print(x)
    #
    #
    #
    #
    #
    # test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    # test_idx_range = np.sort(test_idx_reorder)
    #
    # # if dataset_str == 'citeseer':
    # #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    # #     # Find isolated nodes, add them as zero-vecs into the right position
    # #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    # #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    # #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    # #     tx = tx_extended
    # #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    # #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    # #     ty = ty_extended
    #
    # features = sp.vstack((allx, tx)).tolil()
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    #
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]
    #
    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    #
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    #
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]



    f = open('data/adjMatrix100sim100sen.p', 'rb')
             # r'# 'r' for reading; can be omitted
    adjM=pkl.load(f)

    features =pkl.load(f)
    totalInstances = len(features)

    labels = pkl.load(f)

    adjFull = concatAdjM(adjM, int(totalInstances/nss))
    adj = sp.csr_matrix(adjFull)
    features = sp.vstack((features)).tolil()

    y_train = np.array(labels)
    y_val = np.array(labels)
    y_test = np.array(labels)

    train_mask = np.zeros(totalInstances)
    val_mask = np.zeros(totalInstances)
    test_mask = np.zeros(totalInstances)
    train_size = int(4*totalInstances/5);
    val_size = int(totalInstances/10);
    test_size = int(totalInstances/10);
    for i in range(0, train_size):
        train_mask[i] = 1
    for i in range(train_size, train_size + val_size):
        val_mask[i] = 1
    for i in range(train_size + val_size, test_size + train_size + val_size):
        test_mask[i] = 1

    train_mask = np.array(train_mask, dtype=np.bool)
    val_mask = np.array(val_mask, dtype=np.bool)
    test_mask = np.array(test_mask, dtype=np.bool)





    # a = np.empty(16)
    # b = np.arange(0, 4, 1)
    # ind = np.arange(len(a))
    # np.put(a, ind, b)
    # row=a
    # col=a
    # # data = np.array([1, 0, 0, 0, 0, 1,0,0,1,0,0,1,0,1,1,0])
    # data = np.array([[1, 0, 0, 0], [0, 1,0,0],[1,0,0,1],[0,1,1,0]])
    # # adj = sp.csr_matrix((data, (row, col)), shape=(4, 4))
    # adj= sp.csr_matrix(data)
    # # x = sp.csr_matrix((4, 4), dtype=float)
    # # x = [[0, 1,0,1], [0, 0,0,1], [0, 1,1,1], [1, 1,0,1]]
    # # print(x)
    # # adj = x
    # features = sp.csr_matrix((2, 1), dtype=float)
    # features = [1.0,2.0,4.0,5.0]
    # features = sp.vstack((features)).tolil()
    # y_train = np.array([[0,1],[1,0],[0,1],[1,0]])
    # y_val = np.array([[0,1],[1,0],[0,1],[1,0]])
    # y_test = np.array([[0,1],[1,0],[0,1],[1,0]])
    #
    # train_mask = np.array([1,1,0,0], dtype=np.bool)
    # val_mask = np.array([0,0,1,0], dtype=np.bool)
    # test_mask = np.array([0,0,0,1], dtype=np.bool)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
