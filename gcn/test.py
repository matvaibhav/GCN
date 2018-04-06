import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys



x = sp.csr_matrix((2, 2), dtype=int)
# x=np.matrix([[0, 1, 0, 0],
#            [0, 0, 1, 0],
#            [1, 0, 0, 0]], dtype=int)
x=[[0,0],[0,0]]
print (x)