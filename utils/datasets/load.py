import numpy as np
import pandas as pd
import torch
import torchvision as tv
import os
import csv
import copy
import wilds

from utils.multi_attr_dataset import MultiAttrVisionDataset, StandardMultiAttrWrapper, MultiAttrTensorDataset
from utils.datasets.metadata import known_datasets, get_scaled_resolution


def load(dataset_name, dataset_path, seed=None, train_tx=None, val_tx=None):
  if "torchvision_name" in known_datasets[dataset_name]:
    return load_builtin_tv_dataset(dataset_name, dataset_path, seed, train_tx, val_tx)
  elif "wilds_name" in known_datasets[dataset_name]:
    return load_wilds_datasets(dataset_name, dataset_path, seed, train_tx, val_tx)
  elif dataset_name == "waterbirds":
    return load_waterbirds(dataset_path, train_tx, val_tx)
  elif dataset_name == "waterbirds_variant":
    return load_waterbirds(dataset_path, train_tx, val_tx, modified_version=True)
  elif dataset_name == "treeperson":
    return load_treeperson(dataset_path, train_tx, val_tx)
  elif dataset_name == "cifar10_skew_train":
    return load_builtin_tv_dataset("cifar10", dataset_path, seed, train_tx, val_tx, skew_train=True)
  else:
    raise ValueError(f"{dataset_name} not implemented.")


def get_bit_transforms(original_resolution):
  ##Define image transformations, copied from the original bit_pytorch/train.py
  precrop, crop = get_scaled_resolution(original_resolution)
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  return train_tx, val_tx

def load_builtin_tv_dataset(dataset_name, dataset_path, seed=None, train_tx=None, val_tx=None, skew_train=False):
  if train_tx is None:
    train_tx, _ = get_bit_transforms(known_datasets[dataset_name]["resolution"])
  if val_tx is None:
    _, val_tx = get_bit_transforms(known_datasets[dataset_name]["resolution"])    
  builtin_generator = getattr(tv.datasets, known_datasets[dataset_name]["torchvision_name"])

  if known_datasets[dataset_name]["torchvision_split"] == "train_bool":
    builtin_train = builtin_generator(dataset_path, transform=train_tx, train=True, download=True)
    builtin_train_static = builtin_generator(dataset_path, transform=val_tx, train=True, download=True)
    builtin_test = builtin_generator(dataset_path, transform=val_tx, train=False, download=True)

    if skew_train:
      train_labels = np.array(tuple(y for (x, y, *extra) in builtin_train))
      nc = known_datasets[dataset_name]["num_classes"][0]
      train_label_count = tuple((train_labels==y).sum() for y in range(nc))
      rng = np.random.default_rng() if seed is None else np.random.default_rng(46294623+seed)
      classes_to_filter = rng.choice(range(nc), size=nc//2, replace=False)
      num_samples_wanted = {y: train_label_count[y] for y in range(nc)}
      for y in classes_to_filter:
        num_samples_wanted[y] //= 10
      sub_indices = filter_by_labels(train_labels, num_samples_wanted, seed)
      builtin_train = torch.utils.data.Subset(builtin_train, sub_indices)
      builtin_train_static = torch.utils.data.Subset(builtin_train_static, sub_indices)


    num_val = (len(builtin_train) * 3) // 10
    num_train = len(builtin_train) - num_val
    splits = [num_val, num_train]
    # num_train = num_train // 10 # make train set smaller to speed up the acquisition process
    # splits = [num_val, num_train, len(builtin_train) - num_val - num_train]
    rng = torch.Generator() if seed is None else torch.Generator().manual_seed(23513442+23*seed)
    builtin_train_splits = torch.utils.data.random_split(builtin_train, splits, generator=rng)  
    in_sample_val_set = builtin_train_splits[0]
    train_set = builtin_train_splits[1]

    train_set_static = torch.utils.data.Subset(builtin_train_static, train_set.indices)

    num_val = len(builtin_test) // 10
    num_test = len(builtin_test) - num_val
    splits = [num_val, num_test]
    rng = torch.Generator() if seed is None else torch.Generator().manual_seed(642354352+45*seed)
    builtin_test_splits = torch.utils.data.random_split(builtin_test, splits, generator=rng)
    out_sample_val_set = builtin_test_splits[0]
    test_set = builtin_test_splits[1]
  else:
    raise NotImplementedError("Unknown split option")

  return tuple(StandardMultiAttrWrapper(x, None, known_datasets[dataset_name]["num_classes"][0])
                for x in (train_set, train_set_static, in_sample_val_set, out_sample_val_set, test_set)
              )


def load_wilds_datasets(dataset_name, dataset_path, seed=None, train_tx=None, val_tx=None):
  dataset = wilds.get_dataset(dataset=dataset_name, root_dir=dataset_path, download=True)
  if train_tx is None:
    train_tx, _ = get_bit_transforms(known_datasets[dataset_name]["resolution"])
  if val_tx is None:
    _, val_tx = get_bit_transforms(known_datasets[dataset_name]["resolution"])  

  train_set = dataset.get_subset("train", transform=train_tx)
  train_set_static = dataset.get_subset("train", transform=val_tx)
  # num_train = len(train_set) // 5 # make train set smaller to speed up the acquisition process
  # splits = [num_train, len(train_set) - num_train]
  # rng = torch.Generator() if seed is None else torch.Generator().manual_seed(23513442+23*seed)
  # train1, train2 = torch.utils.data.random_split(train_set, splits, generator=rng)
  # train_set = train1
  # train_set_static = torch.utils.data.Subset(train_set_static, train_set.indices)
  if dataset_name == "fmow":
    # Needed to be dealt with separately because this dataset uses different conventions for some reason
    in_sample_val_set = dataset.get_subset("val", transform=val_tx)
    out_sample_val_set = dataset.get_subset("ood_val", transform=val_tx)
  else:    
    in_sample_val_set = dataset.get_subset("id_val", transform=val_tx)
    out_sample_val_set = dataset.get_subset("val", transform=val_tx)
  test_set = dataset.get_subset("test", transform=val_tx)  

  return tuple(StandardMultiAttrWrapper(x, None, known_datasets[dataset_name]["num_classes"][0])
                for x in (train_set, train_set_static, in_sample_val_set, out_sample_val_set, test_set)
              )


def load_waterbirds(data_path, metadata_path=None, train_tx=None, val_tx=None, modified_version=False):    
  attr_to_idx = {"bird": 0, "place": 1}
  class_to_idx = {a: {"land": 0, "water": 1} for a in attr_to_idx}

  if modified_version:
    ## splits_samples[0,1,2,3] = samples for train, in sample valid, out sample valid, and test sets
    splits_samples = [[] for _ in range(4)]
  else:    
    ## The original waterbirds dataset https://github.com/kohpangwei/group_DR does not have an in sample valid set
    ## splits_samples[0,1,2] = samples for train, (out sample) valid, and test sets    
    splits_samples = [[] for _ in range(3)]
    
  ## build the (sample_path, label) tuples for each train/valid/test split
  if metadata_path is None:
    metadata_path = os.path.join(data_path, "metadata.csv")
  with open(metadata_path, newline='') as csvfile:    
      reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      next(reader)
      for row in reader:
        path = row[1]
        target = torch.empty(2, dtype=torch.int64)
        target[attr_to_idx["bird"]] = int(row[2])
        target[attr_to_idx["place"]] = int(row[4])
        splits_samples[int(row[3])].append((path,target))

  if train_tx is None:
    train_tx, _ = get_bit_transforms(known_datasets["waterbirds"]["resolution"])
  if val_tx is None:
    _, val_tx = get_bit_transforms(known_datasets["waterbirds"]["resolution"])

  if modified_version:
    splits_transforms = [train_tx, val_tx, val_tx, val_tx]
    train_set, in_sample_val_set, out_sample_val_set, test_set = (MultiAttrVisionDataset(
                                                                            data_path,
                                                                            splits_samples[i],
                                                                            attr_to_idx,
                                                                            class_to_idx,
                                                                            transform=splits_transforms[i]
                                                                            ) for i in range(4))
  else:
    splits_transforms = [train_tx, val_tx, val_tx]
    in_sample_val_set = None
    train_set, out_sample_val_set, test_set = (MultiAttrVisionDataset(data_path,
                                                                      splits_samples[i],
                                                                      attr_to_idx,
                                                                      class_to_idx,
                                                                      transform=splits_transforms[i]
                                                                      ) for i in range(3))  
  
  train_set_static = MultiAttrVisionDataset(data_path, splits_samples[0], attr_to_idx, class_to_idx, transform=val_tx)
  return train_set, train_set_static, in_sample_val_set, out_sample_val_set, test_set


def load_treeperson(data_path, train_tx=None, val_tx=None):
  attr_to_idx = {"has_person": 0, "setting": 1}
  class_to_idx = {"has_person": {"no_person": 0, "person": 1}, "setting": {"trees": 0, "buildings": 1}}

  ## This dataset only has a train and an out-of-sample validation set
  splits_samples = [[] for _ in range(2)]

  ## build the (sample_path, label) tuples for each train/valid split
  metadata_path = os.path.join(data_path, "metadata.csv")
  with open(metadata_path, newline='') as csvfile:
      reader = csv.reader(csvfile, delimiter=',', quotechar='|')
      next(reader)
      for row in reader:
        path = row[1]
        target = torch.empty(2, dtype=torch.int64)
        target[attr_to_idx["has_person"]] = int(row[2])
        target[attr_to_idx["setting"]] = int(row[3])
        splits_samples[int(row[4])].append((path, target))

  if train_tx is None:
    train_tx, _ = get_bit_transforms(known_datasets["treeperson"]["resolution"])
  if val_tx is None:
    _, val_tx = get_bit_transforms(known_datasets["treeperson"]["resolution"])

  splits_transforms = [train_tx, val_tx]
  train_set, out_sample_val_set = (MultiAttrVisionDataset(data_path,
                                                          splits_samples[i],
                                                          attr_to_idx,
                                                          class_to_idx,
                                                          transform=splits_transforms[i]
                                                          ) for i in range(2))

  train_set_static = MultiAttrVisionDataset(data_path, splits_samples[0], attr_to_idx, class_to_idx, transform=val_tx)
  return train_set, train_set_static, out_sample_val_set, out_sample_val_set, out_sample_val_set


def filter_by_labels(labels, num_samples_wanted, seed=None):
  """ Args:
  labels (np.array): The labels of a dataset. shape (len(dataset),)
  num_samples_wanted(dict[int]->int): num_samples_wanted[y] = number of samples wanted to select from class y

  Return:
  np.array: Indices that represents a subset with class distribution given by num_samples
  """
  rng = np.random.default_rng() if seed is None else np.random.default_rng(97624937231+seed)
  result = []
  for y, n in num_samples_wanted.items():
    indices_with_given_target = np.nonzero(labels==y)[0]
    select = rng.choice(indices_with_given_target, size=n, replace=False)
    result.extend(select)
  return result

