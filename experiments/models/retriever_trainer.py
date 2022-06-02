import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, DistributedSampler

from feedbackQA.models.optimizer import get_optimizer
from feedbackQA.data.data_loader import DistributedSamplerWrapper
import numpy as np

def soft_cross_entropy(logits, soft_targets):
    probs = torch.nn.functional.log_softmax(logits, dim=1)
    return torch.sum(-soft_targets * probs, dim=1)


def prepare_soft_targets(target_ints, num_classes):
    mask = target_ints == -1
    inverted_mask = torch.logical_not(mask)
    modified_target_ints = inverted_mask * target_ints
    oh_modified_target_ints = one_hot(modified_target_ints, num_classes=num_classes)
    modified_soft_targets = oh_modified_target_ints.double()
    repeated_inverted_mask = inverted_mask.unsqueeze(1).repeat((1, num_classes)).reshape(
        [inverted_mask.shape[0], num_classes])
    soft_targets = (modified_soft_targets * repeated_inverted_mask).float()
    repeated_mask = mask.unsqueeze(1).repeat((1, num_classes)).reshape(
        [mask.shape[0], num_classes])
    uniform_targets = (1 / num_classes) * repeated_mask
    soft_targets += uniform_targets
    return soft_targets


class RetrieverTrainer(pl.LightningModule):

    """
    This class implements the LightningModule from PyTorch Lightning.
    In particular, it defines the methods to to train_step, train_valid.
    See PyTorch Lightning documentation for more info.
    """

    def __init__(self, retriever, train_data, dev_data, test_data, loss_type,
                 optimizer_type, train_source='batch', eval_source='inline', is_distributed=False):
        super(RetrieverTrainer, self).__init__()
        self.retriever = retriever
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.train_source = train_source
        self.eval_source = eval_source
        self.is_distributed = is_distributed

        if loss_type == 'classification':
            self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.val_metrics = {}

    def forward(self, **kwargs):
        return self.retriever(**kwargs)

    def build_candidates(self, batch, source, mode):
        """
        Build a candidate set for this batch.

        :param batch:
            a Batch object 
        :param source:
            the source from which candidates should be built, one of
            ['batch', 'inline', 'fixed']
        :param mode:
            'train' or 'eval'

        :return: tuple of tensors (label_inds, cands, cand_vecs)

            label_inds: A [bsz] LongTensor of the indices of the labels for each
                example from its respective candidate set
            cands: A [num_cands] list of (text) candidates
                OR a [batchsize] list of such lists if source=='inline'
            cand_vecs: A padded [num_cands, seqlen] LongTensor of vectorized candidates
                OR a [batchsize, num_cands, seqlen] LongTensor if source=='inline'

        Possible sources of candidates:

            * batch: the set of all labels in this batch
                Use all labels in the batch as the candidate set (with all but the
                example's label being treated as negatives).
                Note: with this setting, the candidate set is identical for all
                examples in a batch. This option may be undesirable if it is possible
                for duplicate labels to occur in a batch, since the second instance of
                the correct label will be treated as a negative.
            * inline: batch_size lists, one list per example
                If each example comes with a list of possible candidates, use those.
                Note: With this setting, each example will have its own candidate set.
            * fixed: candidate size list, one list valid for all example
                Note: Prefer this setting over inline if all the candidates are same for all examples in batch.
        """

        label_vecs = batch['label']
        label_inds = None
        batchsize = batch['question']['ids'].size(0)

        if label_vecs is not None:
            assert 'ids' in label_vecs.keys()
            assert label_vecs['ids'].dim() == 2

        if source == 'batch':
            # logger.warning(
            #     '[ Executing {} mode with batch labels as set of candidates. ]'
            #     ''.format(mode)
            # )
            # if batchsize == 1:
            #     logger.warning(
            #         "[ Warning: using candidate source 'batch' and observed a "
            #         "batch of size 1. This may be due to uneven batch sizes at "
            #         "the end of an epoch. ]"
            #     )
            # if label_vecs is None:
            #     raise ValueError(
            #         "If using candidate source 'batch', then batch.label_vec cannot be "
            #         "None."
            #     )

            cands = batch['label_text']
            cand_vecs = label_vecs
            old_target = batch['target_idx']
            label_inds = torch.tensor(range(batchsize)).to(old_target.device)

        elif source == 'inline':

            # logger.warning(
            #     '[ Executing {} mode with provided inline set of candidates ]'
            #     ''.format(mode)
            # )
            if batch['passages'] is None:
                raise ValueError(
                    "If using candidate source 'inline', then batch['passages'] "
                    "cannot be None. If your task does not have inline candidates, "
                    "consider using one of --{m}={{'batch','fixed'}}."
                    "".format(m='candidates' if mode == 'train' else 'eval-candidates')
                )

            cands = batch['passages_text']
            cand_vecs = batch['passages']
            if label_vecs is not None:
                label_inds = batch['target_idx']
        
        elif source == 'fixed':

            cands = [item[0] for item in batch['passages_text']]
            cand_vecs = {
                'ids': batch['passages']['ids'][0],
                'am': batch['passages']['am'][0]
            }
            if 'tt' in batch['passages'].keys():
                cand_vecs['tt'] = batch['passages']['tt'][0]
            if label_vecs is not None:
                label_inds = batch['target_idx']

        else:
            raise Exception("Unrecognized source: %s" % source)

        return {
            'question': batch['question'], 
            'passages': cand_vecs
        }, cands, label_inds
    
    def step_helper(self, batch, source, mode):
        inputs, cands, targets = self.build_candidates(batch, source, mode)
        if self.loss_type == 'classification':
            logits = self.retriever.compute_score(**inputs)
            loss = self.cross_entropy(logits, targets)
        elif self.loss_type == 'classification_with_uniform_ood':
            logits = self.retriever.compute_score(**inputs)
            soft_targets = prepare_soft_targets(targets, logits.shape[1])
            loss = soft_cross_entropy(logits, soft_targets)
        else:
            raise ValueError('loss_type {} not supported. Please choose between classification and'
                             ' classification_with_uniform_ood')
        all_prob = self.softmax(logits)
        return loss, all_prob

    def training_step(self, batch, batch_idx):
        train_loss, _ = self.step_helper(batch, self.train_source, 'train')
        # logs
        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def training_step_end(self, outputs):
        loss_value = outputs['loss'].mean()
        tensorboard_logs = {'train_loss': loss_value}
        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataset_number=0):
        # if self.dev_data is a dataloader, there is no provided
        # dataset_number, hence the default value at 0
        loss, predictions = self.validation_compute_predictions(batch)
        targets = batch['target_idx']
        val_acc = torch.tensor(accuracy_score(targets.cpu(), predictions.cpu())).to(targets.device)
        return {'val_loss_' + str(dataset_number): loss, 'val_acc_' + str(dataset_number): val_acc, 'size': targets.size(0)}

    def compute_predictions(self, batch):
        loss, all_prob = self.step_helper(batch, self.eval_source, 'eval')
        _, predictions = torch.max(all_prob, 1)
        return loss, predictions

    def validation_compute_predictions(self, batch, topk=1):
        inputs, cands, targets = self.build_candidates(batch, self.eval_source, 'eval')
        question = self.retriever.embed_tokenized_question(inputs['question'])
        passages = self.retriever.embed_tokenized_paragraphs(inputs['passages'])
        scores, prediction, _ = self.retriever.predict(question, passages, True, True, topk=topk)
        
        if self.loss_type == 'classification':
            loss = self.cross_entropy(scores, targets)
        elif self.loss_type == 'classification_with_uniform_ood':
            soft_targets = prepare_soft_targets(targets, scores.shape[1])
            loss = soft_cross_entropy(scores, soft_targets)
        else:
            raise ValueError('loss_type {} not supported. Please choose between classification and'
                             ' classification_with_uniform_ood')
        return loss, prediction

    def validation_compute_predictions_topk(self, batch, topk=1):
        inputs, cands, targets = self.build_candidates(batch, self.eval_source, 'eval')
        question = self.retriever.embed_tokenized_question(inputs['question'])
        passages = self.retriever.embed_tokenized_paragraphs(inputs['passages'])
        scores, prediction, _ = self.retriever.predict(question, passages, True, True, topk=topk)
        
        return prediction

    def validation_epoch_end(self, outputs):
        """

        :param outputs: if dev is a single dataloader, then this is an object with 2 dimensions:
                        validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.

                        if dev is multiple dataloader, then this is an object with 3 dimensions:
                        dataset, validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.
        :return:
        """

        #  dev_data can be either a single dataloader, or a list of dataloaders
        #  for evaluation on many test sets

        if len(self.dev_data) > 1 and type(self.dev_data) is list:
            # Evaluate all validation sets (if there is more than 1)
            val_metrics = {}
            for dataset_index in range(len(self.dev_data)):
                avg_val_loss = self._comput_mean_for_metric(dataset_index, 'val_loss_', outputs)
                avg_val_acc = self._comput_mean_for_metric(dataset_index, 'val_acc_', outputs)
                val_metrics['val_acc_' + str(dataset_index)] = avg_val_acc
                val_metrics['val_loss_' + str(dataset_index)] = avg_val_loss
        else:  # only one dev set provided
            avg_val_loss = self._comput_mean_for_metric(None, 'val_loss_', outputs)
            avg_val_acc = self._comput_mean_for_metric(None, 'val_acc_', outputs)

            val_metrics = {'val_acc_0': avg_val_acc, 'val_loss_0': avg_val_loss}

        #print([val_metrics['val_acc_{}'.format(i)].cpu().tolist() for i in range(5)])
        val_metrics['val_acc_mean'] = np.mean([val_metrics['val_acc_{}'.format(i)].cpu().tolist() for i in range(5)])
        results = {
            'progress_bar': val_metrics,
            'log': val_metrics
        }
        return results

    def _comput_mean_for_metric(self, dataset_index, metric_name, outputs):
        if dataset_index is not None:
            outputs = outputs[dataset_index]
            metric_index = dataset_index
        else:
            metric_index = 0

        datapoints = [x[metric_name + str(metric_index)] for x in outputs]
        size = [x['size'] for x in outputs]
        if len(datapoints[0].shape) == 0:
            # if just a scalar, create a fake empty dimension for the cat
            datapoints = [dp.unsqueeze(0) for dp in datapoints]
        val_losses = torch.cat(datapoints)
        val_losses = [val_loss * s for val_loss, s in zip(val_losses, size)]
        avg_val_loss = sum(val_losses)/sum(size)
        return avg_val_loss

    def test_step(self, batch, batch_idx):
        # we do the same stuff as in the validation phase
        return self.validation_step(batch, batch_idx)

    def test_end(self, outputs):
        """

        :param outputs: if dev is a single dataloader, then this is an object with 2 dimensions:
                        validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.

                        if dev is multiple dataloader, then this is an object with 3 dimensions:
                        dataset, validation_point, metric_name
                        and it contains N datapoint (tensor with N elements), where N
                        is the number of GPUs.
        :return:
        """

        #  dev_data can be either a single dataloader, or a list of dataloaders
        #  for evaluation on many test sets

    
        avg_val_loss = self._comput_mean_for_metric(None, 'val_loss_', outputs)
        avg_val_acc = self._comput_mean_for_metric(None, 'val_acc_', outputs)

        val_metrics = {'val_acc_0': avg_val_acc, 'val_loss_0': avg_val_loss}

        results = {
            'progress_bar': val_metrics,
            'log': val_metrics
        }
        return results

    def configure_optimizers(self):
        return get_optimizer(self.optimizer_type, self.retriever)

    def train_dataloader(self):
        if self.is_distributed:
            sampler = self.train_data.sampler
            sampler = DistributedSamplerWrapper(sampler, shuffle=False)
            return DataLoader(self.train_data.dataset, sampler=sampler, batch_size=self.train_data.batch_size, num_workers=self.train_data.num_workers)
        return self.train_data

    def val_dataloader(self):
        return self.dev_data

    def test_dataloader(self):
        return self.test_data
