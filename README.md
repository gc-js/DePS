# DePS

The DePS branch contains the implementation of our proposed [DePS](https://arxiv.org/abs/2203.08820) model. This implementation of DePS was transcribed from [DeepNovoV2](https://github.com/volpato30/DeepNovoV2). We improved the feature representation and feature extraction network.

## Dependency
python >= 3.6

pytorch >= 1.0

Cython

dataclasses

## Data files

The ABRF DDA spectrums (the data for Table 1 in the original paper) and the default knapsack.npy file could be downloaded [here](https://drive.google.com/drive/folders/1sS9fTUjcwQukUVCXLzAUufbpR0UjJfSc?usp=sharing).
You can also download the data file from [Baidu net disk](
And the 9 species data could be downloaded [here](ftp://massive.ucsd.edu/MSV000081382/peak/DeepNovo/HighResolution/): ftp://massive.ucsd.edu/MSV000081382/peak/DeepNovo/HighResolution/. 

It is worth noting that
 in our implementation we represent training samples in a slightly different format (i.e. peptide stored in a csv file and spectrums stored in mgf files).
 We also include a script for converting the file format (data_format_converter.py in DePS branch).

## usage
first build cython modules

~~~
make build
~~~

train mode:

~~~
make train
~~~

denovo mode:

~~~
make denovo
~~~

evaluate denovo result:

~~~
make test
~~~


## Troubleshooting

In this section we provide an overview of issues that you may meet and how they were solved.

```shell
1. RuntimeError: CUDA out of memory.
```
Solution:
Change batch_size in deepnovo_config.py

```shell
2. When you run make build you get "./lib/python3.8/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:17:2: warning: #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION".
```
Solution:
open "./lib/python3.8/site-packages/numpy/core/include/numpy/ndarraytypes.h" file, and add "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" to the first line and re-make build
