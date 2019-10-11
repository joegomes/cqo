# cqo
Classical Quantum Optimization

installation:
- conda create -n cqo python=3.7
- source activate cqo
- conda install -y cmake mpich2 numpy
- pip install netket

run examples:
- python optimize.py rudy_8_12_1337.sparse
- python calculate_energy.py rudy_8_12_1337.sparse
- python calculate_timing.py rudy_8_12_1337.sparse
