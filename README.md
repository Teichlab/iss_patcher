# ISS Patcher

`iss_patcher` is a simple package for approximating features not experimentally captured in low-dimensional data based on related, high-dimensional data. The shared feature space between the two objects is identified, and log-normalised and z-scored on a per-object basis. The nearest neighbours of the low-dimensional observations in the high-dimensional space are identified, and the counts of the absent features are approximated as the mean of the high-dimensional neighbours.

While the function was initially written for processing ISS and GEX data, it can in principle be used for any sort of low-dimensional data featuring a subset of features from high-dimensional data.

## System requirements

<details>
<summary><b>show requirements</b></summary>

### Hardware requirements

`iss_patcher` can run on a standard computer with enough RAM to hold the used datasets in memory.

### Software requirements

**OS requirements**

The package has been tested on:

- macOS Monterey (12.6.7)
- Linux: Ubuntu 18.04.6 bionic

**Python requirements**

A python version `>=3.7` and `<3.12` is required for all dependencies to work. 
Various python libraries are used, listed in `pyproject.toml`, including the python scientific stack with `scipy>=1.6.0`, `annoy` and `scanpy`.
`iss_patcher` and all dependencies can be installed via `pip` (see below).

</details>

## Installation

*Optional: create and activate a new conda environment (with python<3.12):*
```bash
mamba create -n iss_patcher "python<3.12"
mamba activate iss_patcher
```

**from github**

```bash
pip install git+https://github.com/Teichlab/iss_patcher.git
```

*(installation time: around 2 min)*

## Usage and Documentation

Please refer to the [demo notebook](notebooks/demo.ipynb). Docstrings detailing the arguments of the various functions can be accessed at [ReadTheDocs](https://iss-patcher.readthedocs.io/en/latest/).

(*demo running time: around 10 min*)

## Citation

`iss_patcher` is part of the forthcoming manuscript "A multiomic atlas of human early skeletal development" by To, Fei, Pett et al. Stay tuned for details!

