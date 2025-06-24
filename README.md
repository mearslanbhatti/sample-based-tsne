# sample-based-tsne

This repository contains notebooks which can be used to compare the Kobak Berens approach of running t-SNE with the multigrid approach.

## Setup

We used python=3.10 for this project. Additionally, the file requirements.txt details the packages that the notebooks require. To setup the venv, make sure you have a conda distribution and follow the steps:

- `conda create -n sample-based-tsne python=3.10`
- `conda activate sample-based-tsne`
- `pip install -r requirements.txt`


## Repository Structure

In the `multigrid-kobakberens-comparison` folder the following notebooks are found:
- `kobak-berens.ipynb`: used to run the kobak and berens approach
- `multi-grid`: used to run the multi-grid approach
- `kobak-berens-tasic`: used to run the kobak and berens approach for the tasic dataset. make sure dataset is available in the `data` diretory.
- `multi-grid-tasic`: used to run multigrid approach on the tasic dataset. make sure dataset is available in the `data` diretory.


## t-SNE implementation used
For the experiments, the FFT-accelerated Interpolation-based t-SNE [implementation](https://github.com/KlugerLab/FIt-SNE).


## Datasets

- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [Tasic et al. Dataset](http://celltypes.brain-map.org/rnaseq)
- [Wong](http://flowrepository.org/id/FR-FCM-ZZTM)
