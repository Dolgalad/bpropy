# Installation

## conda installation
Create a conda environment 
```bash
conda create -n bpropy python=3.10
```

Current implementation depends on `torch-scatter` which requires specific wheel. Install the torch + PyG dependencies
```bash
CUDA=cpu TORCH_VERSION=2.8.0 ./tools/install_pyg.sh
```

If cuda is available replace `CUDA=cpu` with the correct CUDA version

Finally install `bpropy` dependencies with 
```bash
pip install .
```

## Running test suite
The unit tests can be run by executing the following
```bash
python -m unittest discover tests/unit
```
