# Human Body Joints estimation for torch7

Train and test a human body joints estimator network using Lua/Torch7 for single humans on an image. 

This code provides an easy way to train a network on a variety of datasets, all available through the `dbcollection` package. The available datasets for train/test/benchmark are the following:

| Dataset | Train | Test | Benchmark |
| --- | --- | --- | --- |
| [Frames Labeled In Cinema (FLIC)](http://bensapp.github.io/flic-dataset.html)] | Yes | Yes | Yes |
| [Leeds Sports Pose (LSP)](http://www.comp.leeds.ac.uk/mat4saj/lspet.html)] | Yes | Yes | Yes |
| [MPII](http://human-pose.mpi-inf.mpg.de/)] | Yes | Yes | No |
| [MSCOCO](http://mscoco.org/)] | Yes | Yes | No |

> Note: Only FLIC and LSP can be benchmarked in this repo. MPII and MSCOCO datasets have dedicated servers for this purpose.


## Requirements

To run the code in this repository you'll need the following resources:

- [Torch7](http://torch.ch/docs/getting-started.html)
- [dbcollection](https://github.com/farrajota/dbcollection)
- Matlab >= 2012a (for benchmark only)

### Torch7 packages

To use this repo, you should install the following packages:

```lua
luarocks install display
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install tds
luarocks install torchnet
```


### dbcollection package setup/installation

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


#### Data setup

`dbcollection` contains the data annotations necessary to run this code. If the data is not setup, this package will download, extract and process the data annotations to disk. To specify where to store the downloaded/extracted data use the `-data_dir=<path>`. If left empty, and the data will be stored in `~/dbcollection/<dataset>/data/`.

For more information on how to setup a dataset see the [dbcollection](https://github.com/farrajota/dbcollection) repo.


# Getting started

After installing the necessary requirements, it is advised to setup the necessary data before proceeding. Since the code uses `dbcollection` for managing datasets, downloading/setting up the data folders is best if you desire to specify the dataset's directory manually. Then, to start training a network simply do `th train.lua -expID <net_name>`. To use a specific dataset, for example FLIC, specify the `-dataset` input arg when running the script `th train.lua -expID <net_name> -dataset flic`.


Most of the command line options are pretty self-explanatory, and can be found in `options.lua`. The `-expID` option will be used to save important information in a directory like `pose-torchnet/exp/<dataset>/<expID>/`. This directory will include snapshots of the trained model, training/validations logs with loss and accuracy information, and other details of the options set for that particular experiment.


## Training a model

To train a network you simply need to do `th train.lua`. This will train a network with the default parameters. To train a network with other options please see the `options.lua` file or look in `tests/` for some scripts that contain training procedures.


## Testing a model

When training a network, a small sample of the overall dataset is used to compute the current accuracy. To use the entire dataset to compute the total accuracy of a network, run `th test.lua -expID <name_exp> -dataset <name_exp>`. For the MPII dataset, the training set is split into two (train + val) and the evaluation is performed on the `val` set.


## Benchmark

To process the prediction of body joints on a dataset, run `th benchmark.lua  -expID <name_exp> -dataset <name_exp>`. This will process all predictions for all annotations of a dataset and store it to disk to two files (`Predictions.t7` + `Predictions.mat`) in the folder of the experiment provided in `-expID`. 

Furthermore, for the FLIC and LSP datasets, this repo provides a benchmarking procedure to compare the results a trained network with other methods for body joint prediction whose predictions are made available online. For the MPII and MSCOCO datasets, you will need to use their evaluation servers.


## Accuracy metric

For convenience during training, the accuracy function evaluates PCK by comparing the output heatmap of the network to the ground truth heatmap. However, this should provide a good enough performance measure when training a network on any dataset.


## Additional notes

Due to problems in cleaning grad buffers with `:clearState()` in order to store models to disk which would cause crashes due to insufficient memory when re-populating the buffers in the GPU memory, here the model's parameters on GPU are copied to a copy of the model which is stored on Ram. This means that, when the model needs to be loaded, a convertion between `nn` and `cudnn` is required for it to run properly. 

Besides this small catch the network should work after the conversion.


# License

MIT license (see the LICENSE file)


# Acknowledgements

This repo is based on [Newell's](https://github.com/anewell/pose-hg-train) code for training a human pose detector.