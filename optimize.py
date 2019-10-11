import os
import sys
import time
import json
import numpy as np
import netket as nk


def load(path):
    with open(path, 'r') as f:
        data = [line.split() for line in f.read().splitlines()]
    nnodes, _ = [int(i) for i in data.pop(0)]
    Q = np.zeros((nnodes, nnodes))
    for edge in data:
        i, j, Qij = [int(f) for f in edge]
        Q[i-1, j-1] = Qij
        Q[j-1, i-1] = Qij
    return Q


def graph_to_ising(Q):
    J = 0.25*(Q - np.diag(Q.sum(-1)))
    h = np.zeros((Q.shape[0]))
    offset = 0.
    return h, J, offset


def construct_ising_hamiltonian(h, J, offset):
    N = h.shape[0]
    sz = [[1., 0.], [0., -1.]]
    sx = [[0., 1.], [1., 0.]]
    edges = []
    for i in range(N):
        for j in range(i, N):
            if J[i,j] != 0.: edges.append([i, j])
    try:  
        g = nk.graph.CustomGraph(edges)
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        for i in range(N):
            if h[i] != 0.:
                ha += h[i] * nk.operator.LocalOperator(hi, [sz], [[i]])
            for j in range(N):
                if J[i, j] != 0.:
                    ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
    except:
        dense_edges = []
        for i in range(N):
            for j in range(i, N):
                dense_edges.append([i, j])
        g = nk.graph.CustomGraph(dense_edges)
        hi = nk.hilbert.Spin(s=0.5, graph=g)
        ha = nk.operator.LocalOperator(hi, offset)
        for i in range(N):
            if h[i] != 0.:
                ha += h[i] * nk.operator.LocalOperator(hi, [sz], [[i]])
            for j in range(N):
                if J[i, j] != 0.:
                    ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(sz, sz)], [[i, j]])
        
    return g, hi, ha


base_dir = os.path.join(os.getcwd(), 'graphs')
filename = sys.argv[1]
name = filename.split(".")[0]
path = os.path.join(base_dir, filename)
if not os.path.exists(path):
    raise Exception("file %s does not exist." % path)


Q = load(path)
h, J, offset = graph_to_ising(Q)


g, hi, ha = construct_ising_hamiltonian(h, J, offset)
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)
sa = nk.sampler.MetropolisLocal(machine=ma)
op = nk.optimizer.Sgd(learning_rate=0.05)
gs = nk.variational.Vmc(
    hamiltonian=ha,
    sampler=sa,
    optimizer=op,
    n_samples=ma.n_par,
    diag_shift=0.1,
    use_iterative=True,
    method='Sr')
gs.run(output_prefix=name, n_iter=200)
