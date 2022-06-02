import numpy as np
import logging
import random
import math
from operator import itemgetter
from torch.utils.data import BatchSampler, DataLoader, Dataset, DistributedSampler
from torch.utils.data import RandomSampler

from bert_reranker.data.data_loader import ReRankerDataset
from bert_reranker.data.generation_dataloader import GenerationDataset, T5GenerationDataset
from bert_reranker.data.justification_encoding_dataloader import FeedbackEncodingDataset
from bert_reranker.data.reason_dataloader import ReasonDataset

logger = logging.getLogger(__name__)


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        hasher, unhasher = {}, {}
        i = 0
        for dataset in datasets:
            task_id = dataset.get_task_id()
            assert task_id not in task_id_2_data_set_dic, "Duplicate task_id %s" % task_id
            task_id_2_data_set_dic[task_id] = dataset
            for j in range(len(dataset)):
                hasher[(task_id, j)] = i
                unhasher[i] = (task_id, j)
                i = i + 1

        self._task_id_2_data_set_dic = task_id_2_data_set_dic
        self.hasher = hasher
        self.unhasher = unhasher

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = self.unhasher[idx]
        return self._task_id_2_data_set_dic[task_id][sample_id]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, mix_opt, extra_task_ratio, n_gpus=1, sampling_ratios=None):
        self._datasets = dataset._datasets
        self.hasher = dataset.hasher
        self._batch_size = batch_size
        self.n_gpus = n_gpus
        self.largest_dataset_size = max([len(data) for data in dataset._datasets])
        self.sampling_ratios = sampling_ratios if sampling_ratios != None else [len(data) for data in dataset._datasets]  
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio

    def __len__(self):
        return math.ceil(self.largest_dataset_size / (self._batch_size * self.n_gpus)) * self.n_gpus * len(self._datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for cur_dataset in self._datasets:
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
        
        samples_to_grab = self._batch_size
        n_batch = math.ceil(self.largest_dataset_size / (self._batch_size * self.n_gpus)) * self.n_gpus * len(self._datasets)
        multiplicative_factor = n_batch / (self.n_gpus * sum(self.sampling_ratios))
        sampling_numbers = [multiplicative_factor * ratio for ratio in self.sampling_ratios]
        sampling_array = []
        for i, elem in enumerate(sampling_numbers):
            sampling_array += [i] * math.ceil(elem)
        random.shuffle(sampling_array)
        all_indices = []
        for elem in sampling_array:
            all_indices = all_indices + (self.n_gpus * [elem])

        for i, local_task_idx in enumerate(all_indices):
            if i >= n_batch:
                break
            cur_batch_sampler = sampler_iterators[local_task_idx]
            cur_samples = []
            for _ in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_samples.append(cur_sample_org)
                except StopIteration:
                    sampler_iterators[local_task_idx] = samplers_list[local_task_idx].__iter__()
                    cur_batch_sampler = sampler_iterators[local_task_idx]
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_samples.append(cur_sample_org)
            task_id = self._datasets[local_task_idx].get_task_id()
            yield [self.hasher[(task_id, sample_id)] for sample_id in cur_samples]


def generate_dataloader_reranker(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle, keep_ood):
    task_id = 0 ## Not Required for normal training
    dataset = ReRankerDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer, keep_ood)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def generate_dataloader_generation(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle):
    task_id = 0 ## Not Required for normal training
    dataset = GenerationDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def generate_dataloader_reason(data_file, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        num_workers, shuffle, data_class=ReasonDataset):
    task_id = 0 ## Not Required for normal training
    dataset = data_class(data_file, max_question_len, max_paragraph_len, task_id, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def generate_dataloader_multitask(data_files, feedback_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        n_gpus, num_workers, shuffle, keep_ood, sampling_ratios=None):
    datasets = []

    for i, data_file in enumerate(data_files):
        task_id = i
        datasets.append(ReRankerDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer, keep_ood))
    
    for i, feedback_file in enumerate(feedback_files):
        task_id = i + len(data_files)
        datasets.append(GenerationDataset(feedback_file, max_question_len, max_paragraph_len, task_id, tokenizer))
    
    multi_task_train_dataset = MultiTaskDataset(datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(multi_task_train_dataset, batch_size, 0, 0, n_gpus, sampling_ratios)
    return DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, num_workers=num_workers), multi_task_batch_sampler

def generate_dataloader_multi_files(data_files, feedback_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        n_gpus, num_workers, shuffle, keep_ood, sampling_ratios=None, data_class=ReasonDataset):
    datasets = []
    
    for i, feedback_file in enumerate(data_files):
        task_id = i
        datasets.append(data_class(feedback_file, max_question_len, max_paragraph_len, task_id, tokenizer))
    
    multi_task_train_dataset = MultiTaskDataset(datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(multi_task_train_dataset, batch_size, 0, 0, n_gpus, sampling_ratios)
    return DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, num_workers=num_workers), multi_task_batch_sampler


def generate_dataloader_multitask_feedback(data_files, feedback_files, max_question_len, max_paragraph_len, tokenizer, batch_size,
                        n_gpus, num_workers, shuffle, keep_ood, sampling_ratios=None):
    datasets = []

    for i, data_file in enumerate(data_files):
        task_id = i
        datasets.append(ReRankerDataset(data_file, max_question_len, max_paragraph_len, task_id, tokenizer, keep_ood))
    
    for i, feedback_file in enumerate(feedback_files):
        task_id = i + len(data_files)
        datasets.append(FeedbackEncodingDataset(feedback_file, max_question_len, max_paragraph_len, task_id, tokenizer))
    
    multi_task_train_dataset = MultiTaskDataset(datasets)
    multi_task_batch_sampler = MultiTaskBatchSampler(multi_task_train_dataset, batch_size, 0, 0, n_gpus, sampling_ratios)
    return DataLoader(multi_task_train_dataset, batch_sampler=multi_task_batch_sampler, num_workers=num_workers), multi_task_batch_sampler

def generate_dataloader():
    pass

class DatasetFromBatchSampler(Dataset):
    """Dataset of indexes from `Sampler`."""

    def __init__(self, sampler, batch_size):
        """
        Args:
            sampler (Sampler): @TODO: Docs. Contribution is welcome
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.sampler_list = None

    def __getitem__(self, index):
        """Gets element of the dataset.
        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            l = list(self.sampler)
            self.sampler_list = [item for sublist in l for item in sublist]
        return self.sampler_list[index]

    def __len__(self):
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler) * self.batch_size

class DistributedBatchSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        batch_size,
        num_replicas = None,
        rank = None,
        shuffle = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedBatchSamplerWrapper, self).__init__(
            DatasetFromBatchSampler(sampler, batch_size),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.dataset = DatasetFromBatchSampler(self.sampler, self.batch_size)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

