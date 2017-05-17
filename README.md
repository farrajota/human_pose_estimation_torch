# Human Body Joints estimation for torch7

Train and test a human body joints estimator network using Lua/Torch7 for single humans on an image. 

Available datasets for train/test/benchmark:

| Dataset | Train | Test | Benchmark |
| --- | --- | --- | --- |
| [Frames Labeled In Cinema (FLIC)](http://bensapp.github.io/flic-dataset.html)] | Yes | Yes | Yes |
| [Leeds Sports Pose (LSP)](http://www.comp.leeds.ac.uk/mat4saj/lspet.html)] | Yes | Yes | Yes |
| [MPII](http://human-pose.mpi-inf.mpg.de/)] | Yes | Yes | No |
| [MSCOCO](http://mscoco.org/)] | Yes | Yes | No |

> Note: MPII and MSCOCO have dedicated servers for benchmarking.


## Requirements

To run the code in this repository you'll need the following resources:

- [Torch7](http://torch.ch/docs/getting-started.html)
- [Fast R-CNN module](https://github.com/farrajota/fast-rcnn-torch)
- [dbcollection](https://github.com/farrajota/dbcollection)
- Matlab >= 2012a (for benchmark)

### Packages/dependencies installation

To use this example code, some packages are required for it to work: `fastrcnn` and `dbcollection`.


#### fastrcnn

To install the Fast R-CNN package do the following:

- install all the necessary dependencies.

```bash
luarocks install tds
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install torchnet
```

- download and install the package.

```bash
git clone https://github.com/farrajota/fast-rcnn-torch
cd fast-rcnn-torch && luarocks make rocks/*
```

> For more information about the fastrcnn package see [here](https://github.com/farrajota/fast-rcnn-torch).


#### dbcollection

To install the dbcollection package do the following:

- download the git repository to disk.
```
git clone https://github.com/farrajota/dbcollection
```

- install the Python module.
```
cd dbcollection/ && python setup.py install
```

-  install the Lua package.
```
cd APIs/lua && luarocks make
```

> For more information about the dbcollection package see [here](https://github.com/farrajota/dbcollection).


# Getting started

After installing the necessary requirements, it is advised to setup the necessary data before proceeding. Since the code uses `dbcollection` for managing datasets, downloading/setting up the data folders is best if you desire to specify the dataset's directory manually. Then, to start training a network simply do `th train.lua -expID <net_name>`. To use a specific dataset, for example FLIC, specify the `-dataset` input arg when running the script `th train.lua -expID <net_name> -dataset flic`.


Most of the command line options are pretty self-explanatory, and can be found in `options.lua`. The `-expID` option will be used to save important information in a directory like `pose-torchnet/exp/<dataset>/<expID>/`. This directory will include snapshots of the trained model, training/validations logs with loss and accuracy information, and other details of the options set for that particular experiment.


## Data setup

## Training a model

## Testing a model

## Benchmark

## Accuracy metric


# License

MIT license (see the LICENSE file)


# Acknowledgements

This repo is based on [Newell's](https://github.com/anewell/pose-hg-train) code for training a human pose detector.