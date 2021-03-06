# Path Kernel

This codebase has been built on top of the open source, publicly available code base available at: https://github.com/ganguli-lab/Synaptic-Flow

First: Please read the file title README-SynFlow-Base.md for instructions on how to run the code. 

For instructions specific to Path Kernel, please read the following.

## Getting Started
Install all dependencies
```
pip install -r requirements.txt
```
The code was tested with Python 3.7.0.


## Code Base
Below is a description of the major sections of the code base. Run `python main.py --help` for a complete description of flags and hyperparameters.

### Datasets
This code base supports the following datasets: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, ImageNet. 

All datasets except Tiny ImagNet and ImageNet will download automatically.  For Tiny ImageNet, download the data directly from [https://tiny-imagenet.herokuapp.com](https://tiny-imagenet.herokuapp.com), move the unzipped folder ``tiny-imagnet-200`` into the ```Data``` folder, run the script `python Utils/tiny-imagenet-setup.py` from the home folder. For ImageNet setup locally in the ```Data``` folder.

### Models

There are four model classes each defining a variety of model architectures:
 - Default models support basic dense and convolutional model.
 - Lottery ticket models support VGG/ResNet architectures based on [OpenLTH](https://github.com/facebookresearch/open_lth).

### Layers

Custom dense, convolutional, batchnorm, and residual layers implementing masked parameters can be found in the `Layers` folder.

### Pruners

All pruning algorithms are implemented in the `Pruners` folder.

### Experiments

Below is a list and description of the experiment files found in the `Experiment` folder:
 - `prune_only.py`: Used to run the pruning experiments for the Path Kernel Experiments listed here


## Results/Figures

All data used to generate the figures in our paper can be found in the `Results/pruned/` folder.  Run the notebook `Notebooks/Plots - Path Kernel, Weight Movement Model Output Logits.ipynb` to generate the figures.

