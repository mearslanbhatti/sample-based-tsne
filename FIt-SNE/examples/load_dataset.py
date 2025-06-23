from enum import Enum, auto
from pathlib import Path
import os
import gzip
import pickle

import fcsparser
from scipy import sparse
import numpy as np
import pandas as pd
# from src.utils import rnaseqTools


def get_mnist(path, kind="all"):
    """
    TODO Docstring for get_mnist.
    """
    path_to_data = Path(path)
    if not path_to_data.exists():
        raise Exception("mnist data was not found at {}".format(path_to_data))

    labels_path_train = os.path.join(path_to_data, 'train-labels-idx1-ubyte.gz')
    labels_path_test = os.path.join(path_to_data, 't10k-labels-idx1-ubyte.gz')
    images_path_train = os.path.join(path_to_data, 'train-images-idx3-ubyte.gz')
    images_path_test = os.path.join(path_to_data, 't10k-images-idx3-ubyte.gz')

    labels_dict = dict()
    images_dict = dict()

    if kind == 'all' or kind == 'train':
        with gzip.open(labels_path_train, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["train"] = br

    if kind == 'all' or kind == 'test':
        with gzip.open(labels_path_test, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["test"] = br

    if kind == 'all' or kind == 'train':
        with gzip.open(images_path_train, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["train"]), 784)
            images_dict["train"] = images

    if kind == 'all' or kind == 'test':
        with gzip.open(images_path_test, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["test"]), 784)
            images_dict["test"] = images

    labels = np.concatenate(list(labels_dict.values()), axis=0)
    images = np.concatenate(list(images_dict.values()), axis=0)

    return images, labels


def get_fmnist(path, kind="all"):
    """
    TODO Docstring for get_fmnist.
    """
    path_to_data = Path(path)
    if not path_to_data.exists():
        raise Exception("fmnist data was not found at {}".format(path_to_data))

    labels_path_train = os.path.join(path_to_data, 'train-labels-idx1-ubyte.gz')
    labels_path_test = os.path.join(path_to_data, 't10k-labels-idx1-ubyte.gz')
    images_path_train = os.path.join(path_to_data, 'train-images-idx3-ubyte.gz')
    images_path_test = os.path.join(path_to_data, 't10k-images-idx3-ubyte.gz')

    labels_dict = dict()
    images_dict = dict()

    if kind == 'all' or kind == 'train':
        with gzip.open(labels_path_train, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["train"] = br

    if kind == 'all' or kind == 'test':
        with gzip.open(labels_path_test, 'rb') as lbpath:
            br = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            labels_dict["test"] = br

    if kind == 'all' or kind == 'train':
        with gzip.open(images_path_train, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["train"]), 784)
            images_dict["train"] = images

    if kind == 'all' or kind == 'test':
        with gzip.open(images_path_test, 'rb') as imgpath:
            br = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
            images = br.reshape(len(labels_dict["test"]), 784)
            images_dict["test"] = images

    labels = np.concatenate(list(labels_dict.values()), axis=0)
    images = np.concatenate(list(images_dict.values()), axis=0)

    return images, labels


# def get_tasic2018(path):
#     """
#     Dataset used in "The art of using t-SNE for single-cell transcriptomics" paper
#     For information about how to get the data and load it go to the following link:
#     https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb
#     """
#     path_to_data = Path(path)

#     if path_to_data.joinpath("tasic2018.pickle").exists():
#         print("[tasic2018] Pickle found. Loading it.")
#         with open(str(path_to_data.joinpath("tasic2018.pickle")), "rb") as f:
#             tasic2018 = pickle.load(f)
#     else:

#         filename = path_to_data.joinpath("mouse_VISp_gene_expression_matrices_2018-06-14",
#                                          "mouse_VISp_2018-06-14_exon-matrix.csv")
#         counts1, genes1, cells1 = rnaseqTools.sparseload(str(filename))

#         filename = path_to_data.joinpath("mouse_ALM_gene_expression_matrices_2018-06-14",
#                                          "mouse_ALM_2018-06-14_exon-matrix.csv")
#         counts2, genes2, cells2 = rnaseqTools.sparseload(filename)

#         counts = sparse.vstack((counts1, counts2), format='csc')

#         cells = np.concatenate((cells1, cells2))

#         if np.all(genes1 == genes2):
#             genes = np.copy(genes1)

#         filename = path_to_data.joinpath("mouse_VISp_gene_expression_matrices_2018-06-14",
#                                          "mouse_VISp_2018-06-14_genes-rows.csv")
#         genesDF = pd.read_csv(str(filename))
#         ids = genesDF['gene_entrez_id'].tolist()
#         symbols = genesDF['gene_symbol'].tolist()
#         id2symbol = dict(zip(ids, symbols))
#         genes = np.array([id2symbol[g] for g in genes])

#         filename = path_to_data.joinpath("tasic-sample_heatmap_plot_data.csv")
#         clusterInfo = pd.read_csv(str(filename))
#         goodCells = clusterInfo['sample_name'].values
#         ids = clusterInfo['cluster_id'].values
#         labels = clusterInfo['cluster_label'].values
#         colors = clusterInfo['cluster_color'].values

#         clusterNames = np.array([labels[ids == i + 1][0] for i in range(np.max(ids))])
#         clusterColors = np.array([colors[ids == i + 1][0] for i in range(np.max(ids))])
#         clusters = np.copy(ids)

#         ind = np.array([np.where(cells == c)[0][0] for c in goodCells])
#         counts = counts[ind, :]

#         areas = (ind < cells1.size).astype(int)

#         clusters = clusters - 1

#         markerGenes = ['Snap25', 'Gad1', 'Slc17a7', 'Pvalb', 'Sst', 'Vip', 'Aqp4',
#                        'Mog', 'Itgam', 'Pdgfra', 'Flt1', 'Bgn', 'Rorb', 'Foxp2']

#         tasic2018 = {'counts': counts, 'genes': genes, 'clusters': clusters, 'areas': areas,
#                      'clusterColors': clusterColors, 'clusterNames': clusterNames}

#         tasic2018["importantGenesTasic2018"] = rnaseqTools.geneSelection(
#             tasic2018['counts'], n=3000, threshold=32)

#         with open(str(path_to_data.joinpath("tasic2018.pickle")), "wb") as f:
#             pickle.dump(tasic2018, f)

#         print(tasic2018['counts'].shape)
#         print(np.sum(tasic2018['areas'] == 0))
#         print(np.sum(tasic2018['areas'] == 1))
#         print(np.unique(tasic2018['clusters']).size)

#     librarySizes = np.sum(tasic2018['counts'], axis=1)
#     X = np.log2(tasic2018['counts'][:, tasic2018["importantGenesTasic2018"]] / librarySizes * 1e+6 + 1)
#     X = np.array(X)
#     X = X - X.mean(axis=0)
#     U, s, V = np.linalg.svd(X, full_matrices=False)
#     U[:, np.sum(V, axis=1) < 0] *= -1
#     X = np.dot(U, np.diag(s))
#     X = X[:, np.argsort(s)[::-1]][:, :50]

#     return X, tasic2018['clusters']


class CifarVersions(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    CIFAR10_ATSNE = auto()
    CIFAR100_ATSNE = auto()


def load_cifars(version, data_home, return_data_and_labels=True):
    """
    TODO Docstring for load_cifars
    """

    if version == CifarVersions.CIFAR10 or version == CifarVersions.CIFAR10_ATSNE:
        name = 'cifar-10'
    elif version == CifarVersions.CIFAR100 or version == CifarVersions.CIFAR100_ATSNE:
        name = 'cifar-100'
    else:
        raise Exception('Given name does not match "cifar-10" or "cifar-100".')

    full_path = Path.joinpath(Path(data_home), name)

    if version == CifarVersions.CIFAR10 or version == CifarVersions.CIFAR100:

        images_path_train = Path.joinpath(full_path, "train")
        images_path_test = Path.joinpath(full_path, "test")

        images_arr = []
        labels_arr = []

        with open(images_path_train, 'rb') as fo:
            dict_train = pickle.load(fo, encoding='bytes')
            images_arr.append(np.array(dict_train[b'data']))
            labels_arr.append(np.array(dict_train[b'coarse_labels']))

        with open(images_path_test, 'rb') as fo:
            dict_train = pickle.load(fo, encoding='bytes')
            images_arr.append(np.array(dict_train[b'data']))
            labels_arr.append(np.array(dict_train[b'coarse_labels']))

        labels = np.concatenate(labels_arr, axis=0)
        images = np.concatenate(images_arr, axis=0)

    elif version == CifarVersions.CIFAR10_ATSNE or version == CifarVersions.CIFAR100_ATSNE:

        labels = np.loadtxt(str(Path.joinpath(full_path, "atsne_label.txt")), delimiter=" ")
        labels = labels.astype(int)
        images = np.loadtxt(str(Path.joinpath(full_path, "atsne_data.txt")), skiprows=1, delimiter=" ")

    else:
        raise Exception("Given version does not match cifar-10 or cifar-100.")

    if return_data_and_labels:
        return images, labels
    else:
        return images


class ImagenetVersions(Enum):
    IMAGENET_H = auto()
    IMAGENET_M = auto()


def load_imagenets(version, data_home, return_data_and_labels=True):
    """
    Loads different versions of the MNIST dataset. The function was taken from
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py

    Parameters
    __________
    version : Datasets, optional
        Version of the MNIST dataset to load:
        If Datasets.MNIST then regular MNIST is loaded.
        If Datasets.F_MNIST then the fashion MNIST is loaded.
    data_home : str, optional
        Locations of the folder where the datasets are stored.
    return_X_y: bool, optional
        If True, method only returns tuple with the data and its labels.
    kind: str, optional
        Defines if the training set (60000 points) or the test set (10000)
        is loaded.
    """

    if version == ImagenetVersions.IMAGENET_H:
        name = 'imageNet/raw_head0_bottleneck.bin'
        shape = (100000, 128)
    elif version == ImagenetVersions.IMAGENET_M:
        name = 'imageNet/raw_mixed3a.bin'
        shape = (100000, 256)
    else:
        raise Exception('Given version does not match the supported data sets.')

    full_path = os.path.join(Path(data_home), name)

    with open(full_path, 'rb') as p:
        feat = np.frombuffer(p.read(), dtype=np.float32, offset=0)
        feat = feat.reshape(shape)

    labels = np.ones(feat.shape[0])

    if return_data_and_labels:
        return feat, labels
    else:
        return feat


def load_amazon3m(path, return_data_and_labels=True):
    """
    TODO Docstring for load_amazon3m
    """
    with open(str(Path.joinpath(Path(path), "amazon_data.txt")), "rb") as f:
        X = []
        for i, line in enumerate(f):
            if i == 0:
             nrows, ncols = map(int, line.decode("utf-8").split(" "))
            if i > 0:
                X.append(np.fromstring(line, sep=" "))
        X = np.concatenate(X, axis=0).reshape(nrows, ncols)

    labels = np.loadtxt(str(Path.joinpath(Path(path), "amazon_label.txt")))
    labels = labels.astype(int)

    if return_data_and_labels:
        return X, labels
    else:
        return X


def load_flow18(path):
    """
    TODO Docstring
    """
    meta, data = fcsparser.parse(path, meta_data_only=False, reformat_meta=True)
    labels = data["class"]
    data = data[
        ["NIR-CD14-CD19", "BUV395-CD25", "BUV737-CD127", "BUV805-CD8", "PE-Va24", "PE-Cy7-gdTCR", "BV510-CD3",
         "BV605-CD16", "BV786-CD56", "APC-tet", "Alexa-700-CD4"]
    ]
    return data.to_numpy(), labels.to_numpy()


def load_wong(data_home, labels_name=None, return_colors=False, return_numeric_labels=True):
    """
    TODO Docstring
    """

    return_labels = False
    if labels_name is not None and labels_name in ["broad", "organs"]:
        return_labels = True
    
    data_home = Path(data_home)
    print(data_home)
    if not data_home.exists():
        raise Exception("wong data was not found at {}".format(data_home))
        
    path_parsed_csv = data_home.joinpath("10k_parsed.csv")
    if not path_parsed_csv.exists():
        raise Exception("preprocess wong data using `parse_data.R` was not found at {}".format(path_parsed_csv))
    
    path_labels = data_home.joinpath("{}_colors.csv".format(labels_name))
    if not path_labels.exists():
        print("labels path not found labels will not be returned".format(path_labels))
        return_labels = False

    X = pd.read_csv(path_parsed_csv).to_numpy()

    if return_labels:
        labels_df = pd.read_csv(path_labels)
        labels = labels_df[f"{labels_name}_color"] if return_colors else labels_df[f"{labels_name}_name"]
        if return_numeric_labels:
            out_labs = np.zeros(labels.size)
            for i, l in enumerate(np.unique(labels)):
                out_labs[labels == l] = i
            labels = out_labs

    if return_labels:
        return X, labels
    else:
        return X
    

def load_celegans(data_home, return_X_y=True):
    """
    Loads C-ELEGANS data available at https://data.caltech.edu/records/1945 

    Parameters
    __________
    data_home : str, optional
        Locations of the folder where the datasets are stored.
    return_X_y: bool, optional
        If True, method only returns tuple with the data and its labels.
    """    
    import anndata as ad

    # Use default location
    if data_home is None:
        data_home = Path.joinpath(Path(__file__).parent, "datasets")
    else:
        data_home = Path(str(data_home))  # quick fix to deal with incoming os.paths

    full_path = Path.joinpath(data_home, "c_elegans")

    ad_obj = ad.read_h5ad(str(Path.joinpath(full_path, "packer2019.h5ad")))
    X = ad_obj.X

    labels_str = np.array(ad_obj.obs.cell_type)

    _, labels = np.unique(labels_str, return_inverse=True)

    print(labels.shape)

    if return_X_y:
        return X, labels
    else:
        return X