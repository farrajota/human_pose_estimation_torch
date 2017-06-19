# Human Body Joints estimation for torch7

Train and test a human body joints estimator network using Lua/Torch7 for single humans on a single image. This method is a modified version of the original [hourglass networks](https://github.com/anewell/pose-hg-train). For more information see our paper (**TODO: insert link to the paper.**)

This code provides an easy way to train a network on a variety of datasets, all available through the `dbcollection` package. The available datasets for train/test/benchmark are the following:

| Dataset | Train | Test | Benchmark |
| --- | --- | --- | --- |
| [Frames Labeled In Cinema (FLIC)](http://bensapp.github.io/flic-dataset.html) | Yes | Yes | Yes |
| [Leeds Sports Pose (LSP)](http://www.comp.leeds.ac.uk/mat4saj/lspet.html) | Yes | Yes | Yes |
| [MPII](http://human-pose.mpi-inf.mpg.de/) | Yes | Yes | **No**** |
| [MSCOCO](http://mscoco.org/) | Yes | Yes | **No**** |

> Note**: Only the FLIC and LSP datasets are evaluated/benchmarked here. The MPII and COCO datasets have dedicated servers for this purpose on their websites.


##  Network architecture

The network model used here for human body joint estimation is an enhanced version of [Newell's](https://github.com/anewell/pose-hg-train) method described in [his paper](http://arxiv.org/abs/1603.06937) with several modifications:

- replaced ReLUs with [RReLUs](https://github.com/torch/nn/blob/master/doc/transfer.md#rrelu)
- use of [spatialdropout](https://github.com/torch/nn/blob/master/doc/simple.md#spatialdropout) between convolutions
- more data augmentation (more rotation, scaling, colour jittering)
- use of wider feature maps (more kernels) as the image resolution decreases
- replaced rmsprop optimization with [adam](https://arxiv.org/abs/1412.6980)
- additional tweaks to the basic auto-encoder network (**TODO: add representative figure.**)



## Results

### FLIC dataset results

#### PCK(0.2) - Observer Centric

| Method | Elbow | Wrist | Total |
| --- | --- | --- | --- |
| Sapp et al., CVPR'13  | 72.5 | 54.5 | 63.5 |
| Chen et al., NIPS'14  | 89.8 | 86.8 | 88.3 |
| Yang et al., CVPR'16  | 91.6 | 88.8 | 90.2 |
| Wei et al., CVPR'16  | 92.5 | 90.0 | 91.3 |
| Newell et al., arXiv'16  | 98.0 | 95.5 | 96.8 |
| *Ours*  | **98.3** | **96.0** | **97.2** |


### LSP dataset

#### PCK(0.2) - Person Centric

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee  | Ankle | Total |
| --- | --- | --- | --- | --- | --- | ---  | --- | --- |
| Wang et al., CVPR'13  | 84.7 | 57.1  | 43.7  | 36.7  | 56.7  | 52.4 | 50.8 | 54.6 |
| Pishchulin et al., ICCV' 13  | 87.2 | 56.7  | 46.7  | 38.0  | 61.0  | 57.5 | 52.7 | 57.1 |
| Tompson et al., NIPS'14  | 90.6 | 79.2  | 67.9  | 63.4  | 69.5  | 71.0 | 64.2 | 72.3 |
| Fan et al., CVPR'15  | 92.4 | 75.2  | 65.3  | 64.0  | 75.7  | 68.3 | 70.4 | 73.0 |
| Chen et al., NIPS'14  | 91.8 | 78.2  | 71.8  | 65.5  | 73.3  | 70.2 | 63.4 | 73.4 |
| Yang et al., CVPR'16  | 90.6 | 78.1  | 73.8  | 68.8  | 74.8  | 69.9 | 58.9 | 73.6 |
| Rafi et al., BMVC'16  | 95.8 | 86.2  | 79.3  | 75.0  | 86.6  | 83.8 | 79.8 | 83.8 |
| Yu et al., ECCV'16  | 87.2 | 88.2  | 82.4  | 76.3  | 91.4  | 85.8 | 78.7 | 84.3 |
| Belagiannis et al., arXiv'16  | 95.2 | 89.0  | 81.5  | 77.0  | 83.7  | 87.0 | 82.8 | 85.2 |
| Lifshitz et al., ECCV'16  | 96.8 | 89.0  | 82.7  | 79.1  | 90.9  | 86.0 | 82.5 | 86.7 |
| Pishchulin et al., CVPR'16  | 97.0 | 91.0  | 83.8  | 78.1  | 91.0  | 86.7 | 82.0 | 87.1 |
| Insafutdinov et al., ECCV'16  | 97.4 | 92.7  | 87.5  | 84.4  | 91.5  | 89.9 | 87.2 | 90.1 |
| Wei et al., CVPR'16  | **97.8** | 92.5  | 87.0  | 83.9  | 91.5  | 90.8 | 89.9 | 90.5 |
| Bulat et al., ECCV'16  | 97.2 | 92.1  | 88.1  | 85.2  | **92.2**  | 91.4 | 88.7 | 90.7 |
| *Ours*  | 97.7 | **93.0**  | **88.9**  | **85.5**  | 91.5  | **92.0** | **92.1** | **91.5** |


> Note: The network was trained with data from the MPII and LSPe datasets just like most of the methods on the benchmark.

## Installation

### Requirements

To run the code in this repository you'll need the following resources:

- [Torch7](http://torch.ch/docs/getting-started.html)
- Matlab >= 2012a (for running the benchmark code)
- Python >= 2.7 or >= 3.5 (for using  [dbcollection](https://github.com/farrajota/dbcollection))
- NVIDIA GPU with compute capability 3.5+ (6GB+ ram)

> Note: Here we used two 6GB ram GPUs to train the network. For using this code, we recommend, at least, one GPU with 6GB ram for inference and one 12GB ram GPU for training a model.

Also, you'll need to install the following packages:

```lua
luarocks install display
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install tds
luarocks install torchnet
```


### dbcollection

To install the dbcollection package do the following:

- install the Python module.
```
pip install dbcollection
```

- download the git repository to disk.
```
git clone https://github.com/farrajota/dbcollection
```

- install the Lua package.
```
cd APIs/lua && luarocks make
```

> For more information about the dbcollection package see [here](https://github.com/farrajota/dbcollection).



#### Data setup

`dbcollection` contains the data annotations necessary to run this code. If the data is not setup, this package will download, extract and process the data annotations to disk. To specify where to store the downloaded/extracted data use the `-data_dir=<path>`. If left empty, and the data will be stored in `~/dbcollection/<dataset>/data/`.

For more information on how to setup a dataset see the [dbcollection](https://github.com/farrajota/dbcollection) repo.


## Getting started


After installing the necessary requirements, it is advised to setup the necessary data before proceeding. Since the code uses `dbcollection` for managing datasets, downloading/setting up the data folders is best if you desire to specify the dataset's directory manually. Then, to start training a network simply do `th train.lua -expID <net_name>`. To use a specific dataset, for example FLIC, specify the `-dataset` input arg when running the script `th train.lua -expID <net_name> -dataset flic`.


Most of the command line options are pretty self-explanatory, and can be found in `options.lua`. The `-expID` option will be used to save important information in a directory like `pose-torchnet/exp/<dataset>/<expID>/`. This directory will include snapshots of the trained model, training/validations logs with loss and accuracy information, and other details of the options set for that particular experiment.

### Download/Setting up this repo

To use this code, clone the repo into your home directory:

```
git clone --recursive https://github.com/farrajota/human_pose_estimation_torch
```

If you clone the repo into a different directory, please make sure you modify `projectdir.lua` and point to the new path before using the code.

### Training a model

To train a network you simply need to do `th train.lua`. This will train a network with the default parameters. To train a network with other options please see the `options.lua` file or look in `tests/` for some scripts that contain training procedures.


### Testing a model

When training a network, a small sample of the overall dataset is used to compute the current accuracy. To use the entire dataset to compute the total accuracy of a network, run `th test.lua -expID <name_exp> -dataset <name_exp>`. For the MPII dataset, the training set is split into two (train + val) and the evaluation is performed on the `val` set.


### Benchmarking against other methods

To process the prediction of body joints on a dataset, run `th benchmark.lua  -expID <name_exp> -dataset <name_exp>`. This will process all predictions for all annotations of a dataset and store it to disk to two files (`Predictions.t7` + `Predictions.mat`) in the folder of the experiment provided in `-expID`.

Furthermore, for the FLIC and LSP datasets, this repo provides a benchmarking procedure to compare the results a trained network with other methods for body joint prediction whose predictions are made available online. For the MPII and MSCOCO datasets, you will need to use their evaluation servers.

### Additional information

#### Accuracy metric

For convenience during training, the accuracy function evaluates PCK by comparing the output heatmap of the network to the ground truth heatmap. However, this should provide a good enough performance measure when training a network on any dataset.


#### Additional notes

Due to problems in cleaning grad buffers with `:clearState()` in order to store models to disk which would cause crashes due to insufficient memory when re-populating the buffers in the GPU memory, here the model's parameters on GPU are copied to a copy of the model which is stored on Ram. This means that, when the model needs to be loaded, a convertion between `nn` and `cudnn` is required for it to run properly.

Besides this small catch the network should work after the conversion.


## License

MIT license (see the LICENSE file)


## Acknowledgements

This repo is based on [Newell's](https://github.com/anewell/pose-hg-train) code for training a human pose detector.