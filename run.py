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
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
import os
import time
import random
import math
from os.path import join as pjoin  # pylint: disable=g-importing-member
from datetime import datetime

import numpy as np
import torch
import torchvision as tv

import utils
import utils.logger
import utils.lbtoolbox as lb
import utils.multi_attr_dataset as mads
import utils.train_test_common as train_test_common
import utils.datasets.load as dsload
import utils.datasets.metadata as metadata

import models.architecture as architecture
import models.hyper_params as hyper_params
import models.acquisition as acq

def make_datasets(dataset_name, dataset_path, target_attr, logger, seed=None, train_tx=None, val_tx=None):  
  logger.info("Loading datasets...")
  train_set, train_set_static, in_sample_val_set, out_sample_val_set, test_set = dsload.load(dataset_name, dataset_path, seed, train_tx, val_tx)
  
  train_sets = {"full": train_set, "static": train_set_static}
  valid_sets = {"in_sample": in_sample_val_set, "out_sample": out_sample_val_set}
  return train_sets, valid_sets



def active_train(model,
                 acquisitor,
                 num_query_total, query_schedule,
                 train_set, attr, train_batch_size, train_batch_split, num_workers,
                 optim, base_lr, device, chrono, logger, early_stopper,
                 valid_sets, eval_every_acq_round=1,
                 seed_set=[],
                 temp_path_to_save_fresh_model=None):
  
  logger.info("Starting active training...")
  logger.info(f"Full training set size: {len(train_set)}")  
  assert (len(acquisitor.dataset) == len(train_set))
  pool_size = len(train_set) if acquisitor.down_sample is None else acquisitor.down_sample
  logger.info(f"Pool size for each acquisition: {pool_size}")

  acquired_examples = seed_set.copy()
  overall_accuracy_over_time = {key: {} for key in valid_sets}
  group_accuracy_over_time = {key: {} for key in valid_sets}

  ## Saving the initial state of the model, which we return to at the beginning of every acquisition loop
  if temp_path_to_save_fresh_model is None:
    temp_path_to_save_fresh_model = "temp_initial_model_" + datetime.now().strftime("%F_%H%M%S") + ".pth"
  torch.save(model.state_dict(), temp_path_to_save_fresh_model)

  acq_round = 0
  while True:
    # Load fresh model
    logger.info(f"Resetting model to initial state...")
    model.load_state_dict(torch.load(temp_path_to_save_fresh_model))    
    # Train from acquired_examples
    active_train_set = mads.Subset(train_set, acquired_examples)
    with chrono.measure("active learning"):
      train_test_common.train(model, active_train_set, attr, train_batch_size,
                              train_batch_split, num_workers, optim, base_lr, device,
                              chrono, logger, early_stopper)
    # Perform testing every so often
    if acq_round % eval_every_acq_round == 0:
      for key in valid_sets:
        logger.info(f"Evaluating on {key} validation set...")
        results = train_test_common.test(model,
                                         valid_sets[key],
                                         attr,
                                         [1],
                                         train_batch_size,
                                         num_workers,
                                         device,
                                         chrono,
                                         logger)
        overall_accuracy_over_time[key][len(acquired_examples)] = results[1][1]
        group_accuracy_over_time[key][len(acquired_examples)] = results[3][1]
    # Perform acquisition
    acq_round += 1    
    num_query = min(next(query_schedule), num_query_total-(len(acquired_examples)-len(seed_set)))
    if num_query > 0:
      acquisitor.set_model(model)
      logger.info(f"Acquiring {num_query} more examples...")
      with chrono.measure("acquisition"):
        indices = acquisitor(num_query, acquired_examples)
        acquired_examples.extend(indices)
      logger.info(f"Now training on {len(acquired_examples)} examples: {acquired_examples}")
    else:
      break
  os.remove(temp_path_to_save_fresh_model)
  return overall_accuracy_over_time, group_accuracy_over_time, acquired_examples


def main(args):
  if args.seed is not None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

  # Set up device, chrono, logger
  torch.backends.cudnn.benchmark = True  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  chrono = lb.Chrono()
  logger = utils.logger.setup_logger(args)
  logger.info(f"Device is {device}")
    
  logger.info("Initializing model...")
  ds_meta = metadata.known_datasets[args.dataset]  
  target_attr_idx = ds_meta["attr_to_idx"][args.target_attr]
  num_classes = ds_meta["num_classes"][target_attr_idx]
  model = architecture.KNOWN_MODELS[args.model](head_size=num_classes, zero_head=True)
  is_clip_model = (args.model[:4] == "CLIP")
  is_vit_model = (args.model == "ViT")
  if is_vit_model:
      model.set_device(device)
  if args.no_pretrain:
    if is_clip_model:
      logger.info("no_pretrain mode has not been implemented for CLIP models")
      raise NotImplementedError
    else:
      logger.info("Using randomly initialized weights")
  else:
    if is_clip_model:
      logger.info("Model loaded using CLIP API")
    elif is_vit_model:
      logger.info("Model loaded from Huggingface")
    else:
      logger.info(f"Loading model from {args.model}.npz")
      model.load_from(np.load(f"{args.model}.npz"))

  # Make datasets
  with chrono.measure("make datasets"):
    if is_clip_model:
      train_tx = model.clip_preprocess
      val_tx = model.clip_preprocess
    else:
      train_tx = None
      val_tx = None      
    # train_sets, valid_sets = make_datasets(args.dataset, args.datadir,
    #                                        args.target_attr, args.passive_examples_per_class, logger)
    train_sets, valid_sets = make_datasets(args.dataset, args.datadir, args.target_attr, logger, args.seed, train_tx, val_tx)

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)  
  
  # optim = torch.optim.SGD(model.module.head.conv.parameters(), lr=0.003, momentum=0.9) # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
  # optim = torch.optim.Adam(model.parameters(), lr=args.base_lr, eps=1e-4)
  model = model.to(device)
  optim.zero_grad()
  
  # Set up early stopper
  # Note that we are not using batch split here, no matter what args.batch_split is
  # This is because the validation process does not take as much memory
  ks = [k for k in args.eval_topk if k < num_classes] # can't ask for top-k accuracy if k >= num_classes
  model_save_path = pjoin(args.logdir, args.name, "bit.pth.tar")

  if args.stop_from_train is not None:
    early_stopper = train_test_common.TrainLossEarlyStopper(
                      model, model_save_path,
                      relative_threshold=args.stop_from_train,
                      # max_step = args.active_max_step_each
                    )
  else:
    train_valid_set = valid_sets[args.train_valid_split]
    if train_valid_set is None:
      raise NotImplementedError(f"This dataset does not have an {args.train_valid_split} validation split")
    logger.info(f"The training loops use an {args.train_valid_split} validation set with {len(train_valid_set)} examples.")
    early_stopper = train_test_common.ValidationEarlyStopper(
                      model, model_save_path,
                      args.early_stop_check_every, train_valid_set, args.target_attr, ks,
                      args.batch, args.workers,
                      device, chrono, logger,
                      args.active_max_step_each, args.early_stop_patience
                    )

  # Passive training
  # passive_train(model,
  #               train_sets["few_shot"], args.target_attr, args.batch, args.batch_split, args.workers,
  #              optim, args.base_lr, device, chrono, logger, early_stopper)

  # Set up validation sets
  test_valid_sets = {key: valid_sets[key] for key in args.valid_splits}
  logger.info(f"Running evaluation on {len(args.valid_splits)} validation set(s):")
  for key in args.valid_splits:
    if test_valid_sets[key] is None:
      raise NotImplementedError(f"This dataset does not have an {key} validation split")
    logger.info(f"{key} with {len(test_valid_sets[key])} examples")

  
  # Active training
  acq_func = acq.known_acquisition_functions[args.acq_func]
  acquisitor = acq.Acquisitor(model,
                              acq_func,
                              train_sets["static"],
                              device,
                              args.acq_compute_batch_size,
                              args.workers,
                              args.down_sample,
                              args.seed)
  num_query_total = args.acq_num_query_total
  # Exponentially increasing query schedule
  def exp_query_schedule(initial, multiplier):
    state = initial
    while True:
      yield math.floor(state)
      state *= multiplier
  query_schedule = exp_query_schedule(args.acq_num_query_initial, args.acq_num_query_multiplier)
  if args.stop_from_train is None:
    early_stopper.max_step = args.active_max_step_each #early_stopper can be reused, just need to update parameters

  temp_path_to_save_fresh_model = pjoin(args.logdir, args.name, "temp_initial_model.pth")

  # build a seed set
  if args.seed_set_is_balanced:
    if args.seed_set_size % num_classes != 0:
      raise ValueError(f"Cannot make a balanced seed set: The seed set size ({args.seed_set_size}) is not divisible by the number of classes ({num_classes}).")
    seed_examples_per_class = args.seed_set_size // num_classes
    logger.info(f"Looking for {seed_examples_per_class} examples per class for few shot learning...")
    logger.info(f"This can take a while for some datasets e.g. iwildcam...")
    seed_set = mads.find_few_shot_subset(train_sets["full"], args.target_attr, seed_examples_per_class,seed=args.seed).indices
  else:
    temp_rng = np.random.default_rng(args.seed)
    seed_set = temp_rng.choice(len(train_sets["full"]), size=args.seed_set_size, replace=False).tolist()

  overall_accuracy_over_time, group_accuracy_over_time, acquired_examples = active_train(
    model,
    acquisitor, num_query_total, query_schedule,
    train_sets["full"], args.target_attr, args.batch, args.batch_split, args.workers,
    optim, args.base_lr, device, chrono, logger, early_stopper,
    test_valid_sets, eval_every_acq_round=args.eval_every_acq_round,
    seed_set=seed_set,
    temp_path_to_save_fresh_model=temp_path_to_save_fresh_model,
  )

  # Run evaluation again, collect final results
  overall_loss = {}
  overall_topks_accuracy = {}
  group_loss = {}
  group_topks_accuracy = {}  
  for key in args.valid_splits:
    logger.info(f"Running evaluation on {key} validation set with {len(test_valid_sets[key])} examples...")
    overall_loss[key], overall_topks_accuracy[key], group_loss[key], group_topks_accuracy[key] = train_test_common.test(
      model, test_valid_sets[key], args.target_attr, ks, args.batch, args.workers, device, chrono, logger
    )

  logger.info(f"Timings:\n{chrono}")

  return (
    overall_loss, overall_topks_accuracy, overall_accuracy_over_time,
    group_loss, group_topks_accuracy, group_accuracy_over_time,
    acquired_examples,
    )

if __name__== "__main__":
  parser = hyper_params.init_argparser(architecture.KNOWN_MODELS.keys())
  hyper_params.add_active_train_args(parser)
  args = parser.parse_args()
  (
    overall_loss, overall_topks_accuracy, overall_accuracy_over_time,
    group_loss, group_topks_accuracy, group_accuracy_over_time,
    acquired_examples,
    ) = main(args)
  # do something with results here e.g. print(overall_loss["out_sample"])
  print("Done!")