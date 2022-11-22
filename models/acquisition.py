import torch
import numpy as np

class _DatasetByIndex(torch.utils.data.Dataset):
    """Helper for Acquisitor class below
    """
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        sample, _ = self.dataset[index]
        return sample, index
    def __len__(self):
        return len(self.dataset)

class Acquisitor:    
    def __init__(self, model, acq_func, dataset, device, batch_size=None, num_workers=8, down_sample=None, seed=None):
        """Args:    
            model:
            acq_func (callable): Takes in (model, batch of samples) and returns the acquisition scores (see below)
            dataset (torch.utils.data.Dataset): Contains the samples            
            device (torch.device): The device used to compute acquisition scores
            batch_size (int, optional): The number of samples to be loaded to the device each time.
                                        Default: Load the whole dataset at once
            num_workers (int, optional): To configure the torch.utils.data.DataLoader
            down_sample (int, optional): For each acquisition, randomly down sample the sample pool to the given size 
            seed (int, optional): Seed for the down sampling process
        """
        self.model = model
        self.acq_func = acq_func
        self.dataset = dataset
        self.dataset_by_index = _DatasetByIndex(self.dataset)
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers        
        self.down_sample = down_sample        
        self.rng = np.random.default_rng(seed)
        
    def __call__(self, num_query, exclusions=[]):
        """
        Args:
            num_query (int): The number of samples to be selected from the dataset
            exclusions (list of int): Indices of samples to be excluded from acquisition
        Returns:
            List[int] of length num_query: Contains indices in dataset of the samples that the model selects
        """
        exclusion_set = set(exclusions)
        allowed_indices = [i for i in range(len(self.dataset_by_index)) if i not in exclusion_set]
        dataset = torch.utils.data.Subset(self.dataset_by_index, allowed_indices)

        if self.down_sample is None:
            assert(num_query <= len(dataset))
        else:
            assert(num_query <= self.down_sample <= len(dataset))
            sampling_indices = self.rng.choice(len(dataset), size=self.down_sample, replace=False)
            dataset = torch.utils.data.Subset(dataset, sampling_indices)
        batch_size = len(dataset) if self.batch_size is None else self.batch_size
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=self.num_workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False)
        self.model.eval()
        with torch.no_grad():
            top_scores = []
            top_global_indices = []
            for x, global_indicies in loader:
                x = x.to(self.device, non_blocking=True)                
                scores = self.acq_func(self.model, x)
                assert(len(scores.shape) == 1) # assert that computation of top_indices will be correct
                assert(scores.shape[0] == x.shape[0])

                if x.shape[0] <= num_query:
                    top_scores.append(scores)
                    top_global_indices.append(global_indicies)
                else:
                    batch_top_scores, batch_top_indices = torch.topk(scores, num_query)
                    top_scores.append(batch_top_scores)
                    top_global_indices.append(global_indicies[batch_top_indices])

            top_scores = torch.cat(top_scores)
            top_global_indices = torch.cat(top_global_indices)
            
            _, ids = torch.topk(top_scores, num_query)
        return top_global_indices[ids].tolist()


    def set_model(self, model):
        self.model = model


##### ACQUISITION FUNCTIONS #####
# NOTE: ALL ACQUISITION FUNCTIONS HAVE PROTOTYPE f(model, x) AND ARE EXPECTED
#       TO RETURN A TORCH.TENSOR WITH SHAPE (x.shape[0], )
def random(model, x):
    return torch.rand(x.shape[0])

def entropy(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        logits = torch.nn.Softmax(dim=1)(logits) # put through softmax
        plogp = logits*torch.log(logits) # p*log(p)
        uncertainty = -torch.sum(plogp, dim=1) # uncertainty = entropy = sum(-plogp); entropy up -> more uncertain
    return uncertainty

def minimum_margin(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        logits = torch.nn.Softmax(dim=1)(logits) # put through softmax
        top2, _ = torch.topk(logits, 2, dim=1, largest=True, sorted=True) # compute top two predicted probabilities
        margin = top2[:, 0] - top2[:, 1] # margin = top probability - second highest probability
        uncertainty = -margin # lower margin = higher uncertainty
    return uncertainty

def uncertainty(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        logits = torch.nn.Softmax(dim=1)(logits) # put through softmax
        predict_probs, _ = torch.max(logits, 1) # probability of most likely class for each example
        uncertainty = 1-predict_probs # as prediction prob decreases, uncertainty increases
    return uncertainty

def raw_uncertainty(model, x):
    model.eval()
    with torch.no_grad():        
        logits = model(x)
        top_logits, _ = torch.max(logits, 1)
        uncertainty = -1*top_logits # the lower the top logit is, the more uncertainty increases
    return uncertainty

'''
A dictionary from acquisition function names to acquisition function handles.
When using train_query.py, the user muts specify one of the keys from this dictionary
as an option for --acquisition_f.
'''
known_acquisition_functions = {
    "random": random,
    "entropy": entropy,
    "minimum_margin": minimum_margin,
    "uncertainty": uncertainty,
    "raw_uncertainty": raw_uncertainty
}
