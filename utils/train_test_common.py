"""Common operations used in both passive and active learning
"""
import random
import torch
import numpy as np
import time
import itertools

import models.hyper_params as hyper_params
import utils.lbtoolbox as lb

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]

def get_losses_and_topks_results(model, x, y, attr_idx, ks):
  """
  Args:
    x: torch.tensor of shape (batch_size, num_classes)
    y: torch.tensor of shape (batch_size, )
  Returns:
    losses: torch.tensor of shape (batch_size, )
    topks:  list of torch.tensors of shape (batch_size, ), len(topks) = len(ks)
  """
  model.eval()
  with torch.no_grad():
    logits = model(x) ##+ 1e-4 ##adding 1e-4 in half precision mode for numerical stability
    classes = y[:,attr_idx,...]
    losses = torch.nn.functional.cross_entropy(logits, classes, reduction='none')
    topks = topk(logits, classes, ks=ks)
  return losses, topks

class ValidationEarlyStopper:
  """Use validation loss to check early stop condition in the training loops

  Args:
    model
    model_save_path (str)
    check_every (int): run evaluation every so many steps and check for early stopping
    valid_set: The validation dataset used to check for early stop
    attr (str): the target attribute used to compute loss and accuracy (e.g. "bird" for the waterbirds dataset)
    ks (tuple(int)): for each k in ks, compute the accuracy of the top-k predictions
    batch_size (int): used to configure the DataLoader for valid_set
    num_workers (int): used to configure the DataLoader for valid_set
    device (torch.device)
    chrono: see utils.lbtoolbox
    logger: see utils.logger
    max_step (int): force early stop after this many steps, no matter what the validation loss is
    max_patience (int): stop training after the model fails to reach a new loss minimum after this many steps
  """
  def __init__(self, model, model_save_path,
               check_every, valid_set, attr, ks,
               batch_size, num_workers,
               device, chrono, logger,
               max_step = 20_000, max_patience=5):    
    self.model = model
    self.model_save_path = model_save_path
    self.check_every = check_every
    self.valid_set = valid_set
    self.attr_idx = valid_set.attr_to_idx[attr]
    self.ks = ks    
    self.device = device    
    self.chrono = chrono
    self.logger = logger
    self.max_step = max_step
    self.max_patience = max_patience

    self.valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers,
                                                    shuffle=False, pin_memory=True, drop_last=False)

    self.valid_losses = []
    self.topks_accuracies = {k: [] for k in self.ks}
    self.best_loss_index = None

    self.train_losses = []

  def __call__(self, step, train_loss=None):
    """Check for early stop.
    Args:
      step (int): the current step number
      train_loss (float): the current train loss, to be logged. If not provided, log as None
    Returns:
      True if the training should be stopped, False otherwise

    Note: after this method is called, the model is probably in evaluation mode.
          Users need to call model.train() manually if necessary
    """
    self.train_losses.append(train_loss)

    if (step >= self.max_step) or (step % self.check_every) == 0:
      self._run_eval()
      steps_since_minimum = len(self.valid_losses)-1 - self.best_loss_index
      if steps_since_minimum >= self.max_patience:
        checkpoint = torch.load(self.model_save_path)        
        self.logger.info(f"Validation loss has not reached a new minimum in {steps_since_minimum} steps.")
        self.logger.info("Reverting to best saved model...")
        self.model.load_state_dict(checkpoint["model"])
        return True
      elif steps_since_minimum == 0:
        torch.save({"model": self.model.state_dict()}, self.model_save_path)

    return step >= self.max_step


  def clear_logs(self):
    """Reset the recorded losses and accuracies.
    Usually use when this EarlyStopper instance is shared between many training runs.
    """
    self.train_losses = []
    self.valid_losses = []
    self.topks_accuracies = {k: [] for k in self.ks}
    self.best_loss_index = None

  def _run_eval(self):
    self.model.eval()
    all_losses = []
    all_topks_results = {k: [] for k in self.ks}
    end = time.time()

    for x, y in self.valid_loader:
      with torch.no_grad():
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        # measure data loading time
        self.chrono._done("eval load", time.time() - end)

        # compute output, measure accuracy and record loss.
        with self.chrono.measure("eval fprop"):
          batch_losses, batch_topks_results = get_losses_and_topks_results(self.model,x,y,self.attr_idx,self.ks)
          all_losses.extend(batch_losses.cpu())  # Also ensures a sync point.          
          for i, k in enumerate(self.ks):
            all_topks_results[k].extend(batch_topks_results[i].cpu())
      # measure elapsed time
      end = time.time()

    loss = np.mean(all_losses)  
    topks_accuracy = {k: np.mean(all_topks_results[k]) for k in self.ks}

    self.logger.info(f"Validation loss {loss:.5f}")
    for k in self.ks:
      self.logger.info(f"top{k} {topks_accuracy[k]:.2%}")
    self.logger.flush()

    self.valid_losses.append(loss)
    for k in self.ks:
      self.topks_accuracies[k].append(topks_accuracy)

    if (self.best_loss_index is None) or loss < self.valid_losses[self.best_loss_index]:
      self.best_loss_index = len(self.valid_losses)-1

    return loss, topks_accuracy


class TrainLossEarlyStopper:
  """ Use train loss to check early stop condition in the training loops

  Args:
    model
    model_save_path (str)
    check_every (int): check train loss every so many steps and check for early stopping
    relative_threshold (float): stop training when current train loss is less than this time the original loss
  """
  def __init__(self, model, model_save_path, check_every=1, relative_threshold=0.001, max_step=float("inf")):
    self.model = model
    self.model_save_path = model_save_path    
    self.check_every = check_every
    self.relative_threshold = relative_threshold
    self.max_step = max_step
    self.train_losses = []

  def __call__(self, step, train_loss):
    """Check for early stop.
    Args:
      model
      step (int): the current step number
      train_loss (float): the current train loss, to be logged.
    Returns:
      True if the training should be stopped, False otherwise
    """

    self.train_losses.append(train_loss)
    if (
      step >= self.max_step or
      ((step % self.check_every == 0) and (train_loss < self.relative_threshold * self.train_losses[0]))
      ):
      torch.save({"model": self.model.state_dict()}, self.model_save_path)
      return True

    return False

  def clear_logs(self):
    """Reset the recorded losses and accuracies.
    Usually use when this EarlyStopper instance is shared between many training runs.
    """
    self.train_losses = []


#########################################
### STANDARD TRAIN LOOP

def seed_dataloader_worker(worker_seed):
  def worker_init_fn(worker_id):
    np.random.seed(worker_seed)
    random.seed(worker_seed)
  return worker_init_fn

def cycle_reseed_loader(dataset, batch_size, num_workers):
  s = torch.initial_seed() % (2**32)
  while True:
    s = (s + 1) % (2**32)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         shuffle=True, pin_memory=True, drop_last=False,
                                         worker_init_fn=seed_dataloader_worker(s),
                                         generator=torch.Generator().manual_seed(s),
                                         )
    for xy in loader:
      yield xy

def train(model,
          train_set, attr, batch_size, batch_split, num_workers,
          optim, base_lr, device, chrono, logger, early_stopper=None):
  ### SET UP LOSS FUNCTION AND DATA LOADER
  def loss_func(x, y):
    logits = model(x) # + 1e-4 ##adding 1e-4 in half precision mode for numerical stability
    classes = y[:,train_set.attr_to_idx[attr],...]
    loss = torch.nn.functional.cross_entropy(logits, classes)
    return loss

  if early_stopper is not None:
    early_stopper.clear_logs()

  micro_batch_size = batch_size // batch_split
  with lb.Uninterrupt() as u:
    logger.info("Starting training...")
    accum_steps = 0
    step = 0
    end = time.time()

    for x, y in cycle_reseed_loader(train_set, micro_batch_size, num_workers):
      model.train()
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)
      # Stop training on interrupts
      if u.interrupted:
        break
      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)      
      # Update learning-rate, including stop training if over.      
      lr = hyper_params.get_lr(step, len(train_set), base_lr)      
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr
      # Forward pass
      with chrono.measure("fprop"):        
        loss = loss_func(x,y)
        loss_num = float(loss.data.cpu().numpy())  # Also ensures a sync point.
      # Accumulate grads
      with chrono.measure("grads"):
        (loss / batch_split).backward()
        accum_steps += 1
      # Write to logger
      accstep = f" ({accum_steps}/{batch_split})" if batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={loss_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()
      # Update params every batch_split microbatches
      if accum_steps == batch_split:
        with chrono.measure("update"):
          optim.step()
          optim.zero_grad()
        step += 1
        accum_steps = 0

        if (early_stopper is not None) and early_stopper(step, loss):
          break
      end = time.time()


def test(model, test_set, attr, ks, batch_size, num_workers, device, chrono, logger):
  model.eval()

  attr_idx = test_set.attr_to_idx[attr]  
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
  num_classes = tuple(len(test_set.classes[a]) for a in test_set.attrs) #need to use test_set.attrs so that attributes appear in order 
  
  all_counts = np.zeros(num_classes)
  all_losses = np.zeros(num_classes)
  all_topks_results = {k: np.zeros(num_classes) for k in ks}
  end = time.time()

  for x, y in test_loader:
    with torch.no_grad():
      indices = tuple(y.numpy().transpose()) #need to convert y to this form due to numpy's index slicing convention

      # compute output, measure accuracy and record loss.
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)      
      chrono._done("eval load", time.time() - end) # measure data loading time      
      with chrono.measure("eval fprop"):
        batch_losses, batch_topks_results = get_losses_and_topks_results(model,x,y,attr_idx,ks)        
        np.add.at(all_counts, indices, 1)
        np.add.at(all_losses, indices, batch_losses.detach().cpu().numpy())
        for i,k in enumerate(ks):
          np.add.at(all_topks_results[k], indices, batch_topks_results[i].detach().cpu().numpy())
    # measure elapsed time
    end = time.time()

  logger.info("Overall statistics:")
  num_examples = all_counts.sum()
  overall_loss = all_losses.sum()/num_examples
  overall_topks_accuracy = {k: all_topks_results[k].sum()/num_examples for k in ks}
  logger.info(f"Validation loss {overall_loss:.5f}")
  for k in ks:
    logger.info(f"top{k} {overall_topks_accuracy[k]:.2%}")

  logger.info("Group statistics:")
  group_loss = all_losses/all_counts  
  group_topks_accuracy = {k: all_topks_results[k]/all_counts for k in ks}

  classes = tuple(test_set.classes[a] for a in test_set.attrs)
  for class_combination in itertools.product(*classes):
    description = (", ").join([f"{c}{a}" for a, c in zip(test_set.attrs, class_combination)])
    class_combination_idx = tuple(test_set.class_to_idx[a][c] for a, c in zip(test_set.attrs, class_combination))
    logger.info(description)    
    logger.info(f"Validation loss {group_loss[class_combination_idx]:.5f}")
    for k in ks:
      logger.info(f"top{k} {group_topks_accuracy[k][class_combination_idx]:.2%}")

  return overall_loss, overall_topks_accuracy, group_loss, group_topks_accuracy
