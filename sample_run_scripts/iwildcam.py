import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np

import run
import models.hyper_params as hyper_params
import models.architecture as architecture
from utils.datasets.metadata import known_datasets

def plot_acc_over_time(results, line_names, saveloc, savename="accuracies", plot_title=None):
  """ Args:
  results (dict[int->np.array]):
    keys represent numbers of samples processed
    values represent accuracies. Shape (len(line_names),)
  line_names (list(str)): Names of the accuracy lines to plot
  saveloc (str): path to the save directory
  savename (str): name of the plot file, without extension
  plot_title (str):
  """  
  if len(results) == 0:
    return
  num_lines = len(line_names)
  time = np.zeros((len(results), ))
  accuracies = np.zeros((len(results), num_lines))
  for ni, num_examples in enumerate(results.keys()):
    time[ni] = num_examples
    accuracies[ni, :] = results[num_examples].reshape(num_lines)
  plt.figure()
  for i in range(len(line_names)):
    plt.plot(time, accuracies[:, i], label=line_names[i])
  plt.legend()
  if plot_title is not None:
    plt.title(plot_title)
  plt.xlabel("Number of examples (seed + acquired)")
  plt.ylabel("Accuracy")
  plt.savefig(os.path.join(saveloc, savename), bbox_inches='tight', pad_inches = 0)
  plt.close()


if __name__== "__main__":
  parser = hyper_params.init_argparser(architecture.KNOWN_MODELS.keys())
  hyper_params.add_active_train_args(parser)
  parser.add_argument("--comment", type=str, required=False, default="")

  #############################################
  ### SETTINGS FOR THE EXPERIMENTS ###
  comment = "test run"

  ### Dataset info
  dataset_name = "iwildcam"
  dataset_path = "/enter/path/here/datasets/wilds"
  group_labels = [str(x) for x in range(182)]

  if known_datasets[dataset_name]["num_attrs"] == 1:
    target_attr = "---"
  num_classes = known_datasets[dataset_name]["num_classes"]

  ### Logging info
  time_stamp = datetime.now().strftime("%F_%H%M%S")
  logdir_base = f"/enter/path/here/logs/{dataset_name}/{time_stamp}/"
  if not os.path.exists(logdir_base):
    os.makedirs(logdir_base)

  ### Active learning setting
  acquisition_functions = ["uncertainty", "random", "entropy", "minimum_margin"]
  seed_set_size = 182
  seed_set_is_balanced=True
  acq_num_query_total = 182*8
  acq_num_query_initial = 20
  acq_num_query_multiplier = 1
  down_sample = 12_000

  ### General training settings
  model = "BiT-M-R50x1"
  no_pretrain = False
  stop_from_train = [0.001]
  # seeds = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151]
  seeds = [1, 11]
  train_batch_size = 32
  train_batch_split = 8
  early_stop_check_every = 10
  acq_compute_batch_size = 32
  eval_every_acq_round = 2
  valid_splits = ["in_sample", "out_sample"]
  base_lr = 0.003

  #######################################
  if dataset_name in ["waterbirds", "treeperson"]:
    # dataset does not have an in sample validation set
    valid_splits = ["out_sample"]

  num_base_settings = (len(seeds), len(acquisition_functions), len(stop_from_train))
 
  overall_accuracies = {vs: np.empty(num_base_settings) for vs in valid_splits}
  overall_accuracies_over_time = {vs: {} for vs in valid_splits}
  group_accuracies = {vs: np.empty(num_base_settings + num_classes) for vs in valid_splits}
  group_accuracies_over_time = {vs: {} for vs in valid_splits}
  acquired_examples = np.empty(num_base_settings + (seed_set_size+acq_num_query_total,), dtype=np.int64)

  for i0, seed in enumerate(seeds):
    for i1, acq_f in enumerate(acquisition_functions):
      for i2, sft in enumerate(stop_from_train):
        run_name = f"{seed}_{acq_f}_{'sftrain' if sft is not None else 'sfvalid'}"
        log_dir = logdir_base + run_name
        flags = [
          f"--model={model}",
          f"--name={run_name}",
          f"--logdir={log_dir}",
          f"--dataset={dataset_name}",
          f"--target_attr={target_attr}",
          f"--datadir={dataset_path}",
          f"--batch={train_batch_size}",
          f"--batch_split={train_batch_split}",
          f"--early_stop_check_every={early_stop_check_every}",
          f"--acq_func={acq_f}",
          f"--acq_compute_batch_size={acq_compute_batch_size}",
          f"--seed_set_size={seed_set_size}",
          f"--acq_num_query_total={acq_num_query_total}",
          f"--acq_num_query_initial={acq_num_query_initial}",
          f"--acq_num_query_multiplier={acq_num_query_multiplier}",
          f"--eval_every_acq_round={eval_every_acq_round}",
          f"--seed={seed}",
          f"--comment={comment}",
          f"--base_lr={base_lr}",
        ]
        for vs in valid_splits:
          flags.extend(["--valid_splits", vs])
        if down_sample is not None:
          flags.append(f"--down_sample={down_sample}")
        if seed_set_is_balanced:
          flags.append("--seed_set_is_balanced")
        if no_pretrain:
          flags.append("--no_pretrain")
        if sft is not None:
          flags.append(f"--stop_from_train={sft}")
        elif dataset_name == "waterbirds":
          flags.append("--train_valid_split=out_sample")

        _, topk_accu, accu_time, _, group_topk_accu, group_accu_time, acquired = run.main(parser.parse_args(flags))

        acquired_examples[i0, i1, i2, :] = acquired
        np.save(logdir_base + "/acquired_examples.npy", acquired_examples)

        for vs in valid_splits:
          overall_accuracies[vs][i0, i1, i2] = topk_accu[vs][1]
          group_accuracies[vs][i0, i1, i2,...] = group_topk_accu[vs][1]

          if not overall_accuracies_over_time[vs]:
            overall_accuracies_over_time[vs] = {str(key): np.empty(num_base_settings) for key in accu_time[vs]}
          for key in accu_time[vs]:
            overall_accuracies_over_time[vs][str(key)][i0, i1, i2] = accu_time[vs][key]

          if not group_accuracies_over_time[vs]:
            group_accuracies_over_time[vs] = {str(key): np.empty(num_base_settings+num_classes) for key in group_accu_time[vs]}
          for key in group_accu_time[vs]:
            group_accuracies_over_time[vs][str(key)][i0, i1, i2,...] = group_accu_time[vs][key]

          np.save(logdir_base + f"/{vs}_overall_accuracies.npy", overall_accuracies[vs])
          np.savez(logdir_base + f"/{vs}_overall_accuracies_over_time.npz", **overall_accuracies_over_time[vs])
          np.save(logdir_base + f"/{vs}_group_accuracies.npy", group_accuracies[vs])
          np.savez(logdir_base + f"/{vs}_group_accuracies_over_time.npz", **group_accuracies_over_time[vs])
        

  for vs in valid_splits:
    compare_overall = {
      int(key): overall_accuracies_over_time[vs][key][:,:,0].mean(axis=0) for key in overall_accuracies_over_time[vs]
    }
    plot_acc_over_time(compare_overall, acquisition_functions, logdir_base, f"compare_{vs}_overall_accuracies_over_time", f"Overall {vs} accuracies over time")

