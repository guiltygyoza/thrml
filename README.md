This is a fork of https://github.com/extropic-ai/thrml.
Two tests are added:
- `tests/test_train_gaussian.py`
- `tests/vmf_tfim.py`
To run:
1. Create a virtual environment and install dependencies
2. `pytest tests/test_train_gaussian.py -v -s` to train a DBM for Gaussian data distribution.
3. `python tests/vmf_tfim.py` to train a RBM with VMC to simulate a 1-D [TFIM](https://en.wikipedia.org/wiki/Transverse-field_Ising_model) at the critical point.
