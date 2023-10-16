# ISS Patcher

`iss_patcher` is a simple package for approximating features not experimentally captured in low-dimensional data based on related, high-dimensional data. The shared feature space between the two objects is identified, and log-normalised and z-scored on a per-object basis. The nearest neighbours of the low-dimensional observations in the high-dimensional space are identified, and the counts of the absent features are approximated as the mean of the high-dimensional neighbours.

While the function was initially written for processing ISS and GEX data, it can in principle be used for any sort of low-dimensional data featuring a subset of features from high-dimensional data.

## Installation

```bash
pip install git+https://github.com/Teichlab/iss_patcher.git
```

## Usage and Documentation

Please refer to the [demo notebook](notebooks/demo.ipynb). There's a function docstring in the source code, which will be rendered on ReadTheDocs once the package goes live.

## Citation

`iss_patcher` is part of the forthcoming manuscript "A multiomic atlas of human early skeletal development" by To, Fei, Pett et al. Stay tuned for details!