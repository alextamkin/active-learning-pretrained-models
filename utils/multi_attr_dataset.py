import torch
import torchvision as tv
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

class MultiAttrVisionDataset(tv.datasets.vision.VisionDataset):
  """A multi-attribute image dataset (i.e. each image has multiple target labels) structure.

  Args:
    root (string): Root directory path of the data
    samples (list): List of (sample_path, label) tuples.
      sample_path (str): relative path (from root) to data files
      label (torch.tensor): contains labels for each attribute.
        Each label shoud have shape (num_attrs, ) and dtype torch.int64
    attr_to_idx (dict[str -> int]): Maps attributes to their indicies (i.e. position in label tensors)
      len(attr_to_idx) = num_attrs
    class_to_idx (dict[str -> dict[str->int]]):
      class_to_idx[X] maps classes in attribute X to the integers representing them in label
      class_to_idx and attr_to_idx should have the same exact set of keys
    loader (callable, optional): a function to load a data file given its path.
    transform (callable, optional): A function that takes in a raw sample and returns a transformed version.
      E.g, ``transforms.RandomCrop`` for images.
    target_transform (callable, optional): A function that takes in the raw target and transforms it.    

  Attributes:
    samples: see above    
    attr_to_idx: see above
    class_to_idx: see above       
    attrs (list[str]): list of attributes, sorted alphabetically
    classes (dict[str->list[str]]): maps each attribute to list of classes in that attribute, sorted alphabetically

  Methods:
    get_targets(): Return [y for x, y in dataset] but avoid the sample transform to improve speed.
  """
  def __init__(self,
               root,
               samples,
               attr_to_idx,
               class_to_idx,
               loader=tv.datasets.folder.default_loader,
               transform=None,
               target_transform=None
               ):    
    ### Check if inputs are valid
    label_size = torch.Size([len(attr_to_idx)])
    for s in samples:
      assert(s[1].shape == label_size)
    assert(len(class_to_idx)==len(attr_to_idx))
    for k in class_to_idx.keys():
      assert(k in attr_to_idx)

    super().__init__(root, transform=transform, target_transform=target_transform)
    self.samples = samples
    self.attr_to_idx = attr_to_idx
    self.attrs = sorted(attr_to_idx.keys())
    self.class_to_idx = class_to_idx
    self.classes = {a: sorted(class_to_idx[a].keys()) for a in self.attrs}
    self.loader = loader

  def __getitem__(self, index):
    """Args:
      index: int

    Return:
      sample: data file at the given index, after self.transformation is applied
      target (torch.tensor of shape num_attrs): labels for the sample
    """
    path, target = self.samples[index]
    sample = self.loader(os.path.join(self.root, path))
    
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return sample, target

  def __len__(self):
    return len(self.samples)

  def get_targets(self):
    pre_transformed_targets = [y for x, y in self.samples]
    if self.target_transform is None:
      return pre_transformed_targets
    else:
      return [self.target_transform(y) for y in pre_transformed_targets]



############################################
### Builder tools ###

class Subset(torch.utils.data.Dataset):
  """Used to take a lazy-evaluation subset of a MultiAttrVisionDataset instance.

  Args:
    dataset (MultiAttrVisionDataset): The source multi-attribute dataset
    indices (list[int]): Indices in the whole set selected for subset
  """
  def __init__(self, dataset, indices):
    self.dataset = dataset
    self.indices = indices
    self.attr_to_idx = dataset.attr_to_idx
    self.attrs = dataset.attrs
    self.class_to_idx = dataset.class_to_idx
    self.classes = dataset.classes

  def __getitem__(self, index):
    return self.dataset[self.indices[index]]

  def __len__(self):
    return len(self.indices)

  def get_targets(self):
    t = self.dataset.get_targets()
    return [t[i] for i in self.indices]


def find_few_shot_subset(dataset, attr, examples_per_class, seed=None):
  """
  Args:
    dataset (MultiAttrVisionDataset)
    attr (str): a key in dataset.attrs
    examples_per_class (int)
    seed (int): if provided, used to make the output deterministic.

  Returns:
    a Subset of dataset consisting of exactly examples_per_class samples per each class in the given attribute
  """
  targets = torch.stack(dataset.get_targets(), dim=0) #shape (num_samples, num_attributes)
  ## Restrict to the given attribute
  attr_idx = dataset.attr_to_idx[attr]
  targets = targets[:,attr_idx] #shape (num_samples,)

  indices = []
  for i in dataset.class_to_idx[attr].values():    
    indices_with_given_target = torch.nonzero(targets==i, as_tuple=True)[0]
    temp_rng = np.random.default_rng(seed)
    select = temp_rng.choice(indices_with_given_target, size=examples_per_class, replace=False)
    indices.extend(select)
  return Subset(dataset, indices)
  

class AttrSlice(torch.utils.data.Dataset):
  """A single-attribute dataset obtained from a multi-attribute dataset by focusing on a single attribute.

  Attributes:  
    classes (list): List of the class names sorted alphabetically.
    class_to_idx (dict): Dict with items (class_name, class_index).
    samples (list): List of (sample path, class_index) tuples    

  Args:
    dataset (MultiAttrVisionDataset): the source multi-attribute dataset
    attr (string): the attribute of dataset to focus on. Must be a key of dataset.attr_to_idx
  """
  def __init__(self, dataset, attr):
    self.dataset = dataset    
    self.attr = attr      

    self.class_to_idx = dataset.class_to_idx[attr]
    self.classes = dataset.classes[attr]
    self.samples = [(sample, target[dataset.attr_to_idx[attr]]) for sample, target in self.dataset.samples]    

  def __getitem__(self, index):
    sample, target = self.dataset[index]
    target = target[self.dataset.attr_to_idx[self.attr]]
    return sample, target

  def __len__(self):
    return len(self.dataset)

  def get_targets(self):
    pre_transformed_targets = self.dataset.get_targets()
    if self.dataset.target_transform is None:
      return [y[self.dataset.attr_to_idx[self.attr]] for y in pre_transformed_targets]
    else:
      return [self.dataset.target_transform(y)[self.dataset.attr_to_idx[self.attr]] for y in pre_transformed_targets]



############################################
### Wrappers for other dataset structures ###

class StandardMultiAttrWrapper:
  """ Wrapper for usual (single-attribute) datasets

  Args:
    dataset: a torch.utils.data.Dataset that returns (sample, label,...), where label is a number (0-dim tensor also ok)
    class_to_idx: See the documentary for MultiAttrVisionDataset.
                  If not provided, a placeholder dictionary will be created using the num_classes argument
    num_classes (int): The number of classes in the dataset.
                       Only required if class_to_idx is not provided.
    transform (callable): To be applied to the sample before returning.
    target_transform (callable): To be applied to the target label before returning.
  Methods:
      get_targets(): Return [y for x, y, ... in dataset] but avoid the sample transform to improve speed
  """
  def __init__(self, dataset, class_to_idx=None, num_classes=None, transform=None, target_transform=None):
    self.dataset = dataset
    self.attr_to_idx = {"---": 0}
    self.attrs = ["---"]
    if class_to_idx is None:
      if num_classes is None:
        raise ValueError("At least one of class_to_idx or num_classes must be provided")
      self.class_to_idx = {"---": {str(x):x for x in range(num_classes)}}
      self.classes = {"---": [str(x) for x in range(num_classes)]}
    else:
      self.class_to_idx = class_to_idx
      self.classes = {a: sorted(self.class_to_idx[a].keys()) for a in self.attrs}
    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    x, y, *extra_stuff = self.dataset[index]
    if self.transform is not None:
      x = self.transform(x)
    if self.target_transform is not None:
      y = self.target_transform(y)    
    if not isinstance(y, torch.Tensor):
      y = torch.tensor(y)
    return x, y.reshape(1)

  def __len__(self):
    return len(self.dataset)

  def get_targets(self):
    placeholder_transform = tv.transforms.Compose([
        tv.transforms.CenterCrop(1),
        tv.transforms.ToTensor()
    ])    
    if hasattr(self.dataset, "transform"):      
      orig_transform = self.dataset.transform
      self.dataset.transform = placeholder_transform      
      temp_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1024, shuffle=False, num_workers=8)
      ys = []
      for x_batch, y_batch, *extra_stuff in temp_loader:
        ys.append(y_batch)
      targets = torch.cat(ys).reshape(-1,1).unbind()
      self.dataset.transform = orig_transform
    elif isinstance(self.dataset, torch.utils.data.Subset) and hasattr(self.dataset.dataset, "transform"):
      # Needed because torch.utils.data.Subset does not inherit attributes from its underlying dataset
      # Could be modified to work for more deeply nested subsets, but this is good for now
      orig_transform = self.dataset.dataset.transform
      self.dataset.dataset.transform = placeholder_transform
      temp_loader = torch.utils.data.DataLoader(self.dataset, batch_size=1024, shuffle=False, num_workers=8)
      ys = []
      for x_batch, y_batch, *extra_stuff in temp_loader:
        ys.append(y_batch)
      targets = torch.cat(ys).reshape(-1,1).unbind()
      self.dataset.dataset.transform = orig_transform
    else:
      print("Warning: This might be very slow")
      targets = torch.tensor([target for sample, target, *extra_stuff in self.dataset]).reshape(-1,1).unbind()
    
    if self.target_transform is not None:      
      targets = [self.target_transform(y) for y in targets]
    return targets


class MultiAttrTensorDataset(torch.utils.data.TensorDataset):
  def __init__(self, tensors, attr_to_idx, class_to_idx):
    super().__init__(*tensors)
    self.attr_to_idx = attr_to_idx
    self.attrs = sorted(attr_to_idx.keys())
    self.class_to_idx = class_to_idx
    self.classes = {a: sorted(class_to_idx[a].keys()) for a in self.attrs}

