# Slightly modified from Google BiT source code (original license below)
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# coding: utf-8

import argparse
import os
import models.architecture as architecture
import models.acquisition as acq

from utils.datasets.metadata import known_datasets

def init_argparser(known_models):  
  parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
  parser.add_argument("--name", type=str, required=True,
                      help="Name of this run. Used for monitoring and checkpointing.")
  parser.add_argument("--model", choices=list(known_models),
                      help="Which variant to use; BiT-M gives best results.")
  parser.add_argument("--logdir", type=str, required=True,
                      help="Where to log training info (small).")
  parser.add_argument("--no_pretrain", action='store_true',
                      help="If this flag is provided, the model will not load pretrained weights.")

  parser.add_argument("--dataset", choices=list(known_datasets.keys()),
                      help="Choose the dataset. It should be easy to add your own! "
                      "Don't forget to set --datadir if necessary.")
  parser.add_argument("--datadir", type=str, required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")

  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--batch", type=int, default=512,
                      help="Batch size for training.")
  parser.add_argument("--batch_split", type=int, default=1,
                      help="Number of batches to compute gradient on before updating weights.")
  parser.add_argument("--base_lr", type=float, default=0.003,
                      help="Base learning-rate for fine-tuning. Most likely default is best.")
  parser.add_argument("--early_stop_check_every", type=int, default=10,
                      help="Check early stop condition after every so many gradient descent steps.")
  parser.add_argument("--eval_topk", type=int, nargs="*", default=[1],
                      help="The k values for which to compute top-k accuracies during evaluation.")

  parser.add_argument("--target_attr", type=str, required=True,
                      help="The attribute we are trying to classify. E.g. 'bird' for 'waterbirds'.")

  parser.add_argument("--train_valid_split", type=str, choices=("in_sample", "out_sample"), default="out_sample",
                      help="The type of validation set used in the training loop for early stopping.")
  #action="append" makes it less convenient to run from command line but easier to run from script
  parser.add_argument("--valid_splits", type=str, action="append", choices=("in_sample", "out_sample"), required=True,
                      help="The types of validation sets used at the end of training. "
                      "Use this flag multiple times to enter multiple validation sets.")

  parser.add_argument("--early_stop_patience", type=int, default=5,
                      help="Stop training if the validation loss has not reached new minima after this many steps")
  parser.add_argument("--stop_from_train", type=float, default=None,
                      help="If provided, training will early stop when train loss reaches this ratio of initial loss. "
                      "In this case, no validation losses are used for early stopping.")
  return parser

def add_active_train_args(parser):
  parser.add_argument("--down_sample", type=int, default=None,
                      help="If provided, randomly down sample the sample pool to the given side for each acquisition")
  parser.add_argument("--acq_func", choices=list(acq.known_acquisition_functions.keys()), default="uncertainty",
                      help="The acquisition function to user during active learning.")
  parser.add_argument("--acq_num_query_total", type=int, default=5,
                      help="The maximum number of samples that can be used in active learning.")
  parser.add_argument("--acq_num_query_initial", type=int, default=1,
                      help="The number of samples to be selected in the first round of active learning.")
  parser.add_argument("--acq_num_query_multiplier", type=float, default=1.0,
                      help="The multiplicative increase rate of the number of samples selected in each acquisition round.")
  parser.add_argument("--acq_compute_batch_size", type=int, default=None,
                      help="The number of samples to compute acquisition scores each time. None means full batch.")
  parser.add_argument("--active_max_step_each", type=int, default=20_000,
                      help="The maximum number of training steps in each loop of active learning.")
  parser.add_argument("--seed_set_size", type=int, default=32,
                      help="Size of the seed set used for initial learning.")
  parser.add_argument("--seed_set_is_balanced", action='store_true',
                      help="If this flag is used, each class will have the same number of examples in the seed set.")
  parser.add_argument("--seed", type=int, default=None,
                      help="Seed used to randomly generate the seed set")
  parser.add_argument("--eval_every_acq_round", type=int, default=1,
                      help="Run prediction on validation set after every so many acquisition rounds.")

def get_mixup(dataset_size):
  return 0.0 if dataset_size < 20_000 else 0.1


def get_schedule(dataset_size):
  if dataset_size < 20_000:
    return [100, 200, 300, 400, 500]
  if dataset_size < 500_000:
    return [500, 3000, 6000, 9000, 10_000]
  else:
    return [500, 6000, 12_000, 18_000, 20_000]


def get_lr(step, dataset_size, base_lr=0.003):
  """Returns learning-rate for `step` or None at the end."""
  supports = get_schedule(dataset_size)
  # Linear warmup  
  if step < supports[0]:
    return base_lr * (step+1) / supports[0]
  # End of training
  elif step >= supports[-1]:
    return None
  # Staircase decays by factor of 10
  else:
    for s in supports[1:]:
      if s < step:
        base_lr /= 10
    return base_lr
