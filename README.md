# cqo
Classical Quantum Optimization

See: https://arxiv.org/abs/1910.10675

installation:
- conda create -n cqo python=3.7
- conda activate cqo
- conda install -y cmake mpich numpy scipy
- pip install netket

run examples:
- python optimize.py rudy_8_12_1337.sparse
- python calculate_energy.py rudy_8_12_1337.sparse
- python calculate_timing.py rudy_8_12_1337.sparse
