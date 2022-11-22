"""Conventions:
known_datasets["dataset_name"] = {
  "resolution": (int, int),
  "num_attrs": int,
  "num_classes": tuple of ints,
  "torchvision_name" (optional): str. Attribute name in torchvision.datasets, if applicable
  "wilds_name" (optional): str. Dataset name in wilds, if applicable
  ...,
}
"""

known_datasets = {}

known_datasets["cifar10"] = {
  "resolution": (32, 32),
  "num_attrs": 1,
  "num_classes": (10,),
  "torchvision_name": "CIFAR10",
  "torchvision_split": "train_bool",
}

known_datasets["cifar10_skew_train"] = {
  "resolution": (32, 32),
  "num_attrs": 1,
  "num_classes": (10,),
}

known_datasets["cifar100"] = {
  "resolution": (32, 32),
  "num_attrs": 1,
  "num_classes": (100,),
  "torchvision_name": "CIFAR100",
  "torchvision_split": "train_bool",
}

known_datasets["caltech101"] = {
  "resolution": (300, 200),
  "num_attrs": 1,
  "num_classes": (101,),
  "torchvision_name": "Caltech101",
  "torchvision_split": "none",
}

known_datasets["waterbirds"] = {
  "resolution": (500, 500),
  "num_attrs": 2,
  "num_classes": (2, 2),
  "attr_to_idx": {"bird": 0, "place": 1},
  "class_to_idx": {a: {"land": 0, "water": 1} for a in ["bird", "place"]},
}

known_datasets["waterbirds_variant"] = {
  "resolution": (500, 500),
  "num_attrs": 2,
  "num_classes": (2, 2),
  "attr_to_idx": {"bird": 0, "place": 1},
  "class_to_idx": {a: {"land": 0, "water": 1} for a in ["bird", "place"]},
}

known_datasets["iwildcam"] = {
  "resolution": (500, 500),
  "num_attrs": 1,
  "num_classes": (182,),
  "wilds_name": "iwildcam",
}

known_datasets["camelyon17"] = {
  "resolution": (96, 96),
  "num_attrs": 1,
  "num_classes": (2,),
  "wilds_name": "camelyon17",
}

known_datasets["fmow"] = {
  "resolution": (224, 224),
  "num_attrs": 1,
  "num_classes": (62,),
  "wilds_name": "fmow",
}

known_datasets["treeperson"] = {
  "resolution": (500, 500),
  "num_attrs": 2,
  "num_classes": (2, 2),
  "attr_to_idx": {"has_person": 0, "setting": 1},
  "class_to_idx": {"has_person": {"no_person": 0, "person": 1}, "setting": {"trees": 0, "buildings": 1}},
}

for ds in known_datasets:
  if known_datasets[ds]["num_attrs"]==1:
    if "attr_to_idx" not in known_datasets[ds]:
      known_datasets[ds]["attr_to_idx"] = {"---": 0}
    if "class_to_idx" not in known_datasets[ds]:
      n = known_datasets[ds]["num_classes"][0]
      known_datasets[ds]["class_to_idx"] = {"---": {str(x):x for x in range(n)}}

def get_scaled_resolution(original_resolution):
  """Takes (H,W) and returns (precrop, crop)."""
  area = original_resolution[0] * original_resolution[1]
  return (160, 128) if area < 96*96 else (512, 480)


# known_dataset_sizes = {
#   'celebA': (1024, 1024),
#   'cifar10': (32, 32),
#   'cifar100': (32, 32),
#   'oxford_iiit_pet': (224, 224),
#   'oxford_flowers102': (224, 224),
#   'imagenet2012': (224, 224),
#   'waterbirds': (500, 500),
#   'waterbirds_places': (500, 500),
# }
