# DePS

The DePS branch contains the implementation of our proposed [DePS](https://arxiv.org/abs/2203.08820) model. This implementation of DePS was transcribed from [DeepNovoV2](https://github.com/volpato30/DeepNovoV2). We improved the feature representation and feature extraction network.

## Dependency
python >= 3.6

pytorch >= 1.0

Cython

dataclasses

## data files

The ABRF DDA spectrums (the data for Table 1 in the original paper) and the default knapsack.npy file could be downloaded [here](https://drive.google.com/drive/folders/1sS9fTUjcwQukUVCXLzAUufbpR0UjJfSc?usp=sharing).
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

