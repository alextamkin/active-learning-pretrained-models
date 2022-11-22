This code is the official PyTorch implementation of the paper *Active Learning Helps Pretrained Models Learn the Intended Task* (https://arxiv.org/abs/2204.08491).

This project mainly uses the Google BiT models. We reuse a lot of code and settings from the official BiT repository (https://github.com/google-research/big_transfer). So far, only the vision tasks are available. The NLP portion of the code will be released later.

## Requirements
See `requirements.txt` for the list of required packages. They can be installed by
```
conda install --file requirements.txt
```
or
```
pip install -r requirements.txt
```

*Note*: The packages `torch-scatter` and `torch-geometric`, which are required for `wilds`, might require a manual installation. See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries for more instructions.

## Datasets
All datasets need to be loaded using the `utils/datasets/load.py` module. The function `load` in this module contains the list of available datasets. To add a new dataset, edit this function and add an entry to the `known_datasets` dictionary in `utils/datasets/metadata.py` .

Below is some information about the default datasets. Note that some of them need to be downloaded manually.

### Waterbirds
This dataset needs to be downloaded manually and is available at https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz 

`utils/dataset/waterbirds` contains a slightly modified version of the original script (https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_waterbirds.py), which can be used to generate variants with different percentages of mis-matched background.

### Treeperson
Created from the GQA dataset (https://cs.stanford.edu/people/dorarad/gqa/index.html). The file `utils/datasets/treeperson/metadata.csv` contains the list of images chosen from the GQA dataset, together with their labels and splits.

This dataset needs to be downloaded manually. Place the GQA images in `/some/path/images/`, then copy `utils/datasets/treeperson/metadata.csv` to `/some/path/metadata.csv` .

### iWildCam2020-WILDS
This dataset will be downloaded automatically if necessary. It is also available at https://wilds.stanford.edu/datasets/

*Note:*

(1) `utils/datasets/load.py` is only compatible with WILDS v1.1 and 1.2. This is because WILDS v2.0 changes the datasets' split dictionaries. A small modification to the function `load_wilds_datasets` in `utils/datasets/load` would be needed to accommodate these changes.

(2) For this dataset, it might take a while for the run scripts above to build a seed set.


## Usage
**Weights:** The model weight file, if required, should be downloaded to the main directory. The Google BiT model weights are available at the official repository linked above. For example, to download the BiT-M-R50x1 model weights, run
```
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```
**Run:** To train a model and print training progress, validation accuracy, etc, run `python3 -m run (flags)`. For example:
```
python3 -m run --name test_run --model BiT-M-R50x1 --logdir /path/to/dir --dataset waterbirds --datadir /datasets/waterbird_complete95_forest2water2/ --target_attr bird --valid_splits out_sample
```
For the list of flags, either run
```
python3 -m run -h
```
or see `models/hyper_params.py`

**Quick start:** It might be more convenient to run experiments from a script. Some sample scripted runs are provided in `sample_run_scripts`. To use them, change the `dataset_path` and `logdir_base` variables to the appropriate paths, then run of the following:
```
python3 -m sample_run_scripts.waterbirds
python3 -m sample_run_scripts.treeperson
python3 -m sample_run_scripts.iwildcam
```

