import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from bert_reranker.models.optimizer import get_optimizer
from bert_reranker.data.multitask_dataloader import DistributedBatchSamplerWrapper

import torch.nn.functional as F
import numpy as np

class RatingTrainer(pl.LightningModule):

    """
    Given the q and a, generate feedback and rating
    """

    def __init__(self, model, train_data, train_sampler, dev_data, loss_type,
                 optimizer_type, train_source='batch', eval_source='inline', is_distributed=False, 
                 pad_id=-1, tokenizer=None, **kwargs):
        super(RatingTrainer, self).__init__()
        self.model = model
        self.train_data = train_data
        self.train_sampler = train_sampler
        self.dev_data = dev_data
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.train_source = train_source
        self.eval_source = eval_source
        self.is_distributed = is_distributed
        self.pad_id = pad_id
        if loss_type == 'classification':
            self.cross_entropy = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.val_metrics = {}
        self.tokenizer = tokenizer
        hsz = self.model.bert_question_encoder.bert.config.d_model
        self.rate_net = nn.Linear(hsz, 4)
        self.rate_loss_func = nn.KLDivLoss()
        self.model.bert_paragraph_encoder = None

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def predict(self, encoder_input):
        decoder_states, _ = self.model.bert_question_encoder.bert(
                                input_ids=encoder_input['ids'], 
                                attention_mask=encoder_input['am'])
                                #token_type_ids=encoder_input['tt'])
        rating_logits = self.rate_net(decoder_states[:, -1, :])
        rating_pred = torch.argmax(rating_logits, dim=-1)
        return rating_pred

    def step_inference(self, encoder_input, *unused):
        decoder_states = self.model.bert_question_encoder.bert(
                                input_ids=encoder_input['ids'].cuda(), 
                                attention_mask=encoder_input['am'].cuda())[0]

        rating_logits = self.rate_net(decoder_states[:, -1, :])
        rating_score = F.softmax(rating_logits, dim=-1)
        rating_score = rating_score[:, -1] 
        return rating_score.argmax(), rating_score, ''

    def step_helper_generator(self, batch):
        encoder_input = batch['pair']
        decoder_input = batch['decoder_input']
        decoder_target = batch['decoder_target']['ids']
        target_rating = batch['mean_rating']
        target_rating_acc = batch['rating']
        
        decoder_states = self.model.bert_question_encoder.bert(
                                input_ids=encoder_input['ids'], 
                                attention_mask=encoder_input['am'])[0]

        rating_logits = self.rate_net(decoder_states[:, -1, :])
        rating_pred = torch.argmax(rating_logits, dim=-1)
        rating_prob = F.log_softmax(rating_logits, dim=-1)
        rating_loss = self.rate_loss_func(rating_prob, target_rating.type(torch.float32))
        rating_acc = pl.metrics.functional.accuracy(rating_pred, target_rating_acc, num_classes=4)

        return rating_loss, rating_acc, rating_pred
    
    def training_step(self, batch, batch_idx):
        rating_loss, rating_acc, rating_pred = self.step_helper_generator(batch)
        tensorboard_logs = {'rating_train_loss': rating_loss, 'generation_train_loss': 0.}
        return {
            'loss': rating_loss, 
            'acc': rating_acc, 
            'rating_loss': rating_loss, 
            'generation_loss': 0.,
            'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataset_number=0):
        # if self.dev_data is a dataloader, there is no provided
        # dataset_number, hence the default value at 0
        #if 'decoder_target' in batch.keys():
        rating_val_loss, rating_val_acc, generation_val_loss, rating_pred = self.step_helper_generator(batch)
        return {'val_loss_' + str(dataset_number): rating_val_loss, 
                'val_acc_' + str(dataset_number): rating_val_acc, 
                #'val_pred_' + str(dataset_number): rating_pred,
                'size': batch['decoder_input']['ids'].size(0)}


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
                val_metrics['val_loss_' + str(dataset_index)] = avg_val_loss
                acc_present = 0
                for x in outputs[dataset_index]:
                    if 'val_acc_' + str(dataset_index) in x.keys():
                        acc_present = 1
                        break
                if acc_present:
                    avg_val_acc = self._comput_mean_for_metric(dataset_index, 'val_acc_', outputs)
                    val_metrics['val_acc_' + str(dataset_index)] = avg_val_acc
        else:  # only one dev set provided
            avg_val_loss = self._comput_mean_for_metric(None, 'val_loss_', outputs)
            val_metrics = {'val_loss_0': avg_val_loss}
            acc_datapoints = [x['val_acc_0'] for x in outputs]
            acc_present = 0
            for x in outputs:
                if 'val_acc_0' in x.keys():
                    acc_present = 1
                    break
            if acc_present:
                avg_val_acc = self._comput_mean_for_metric(None, 'val_acc_', outputs)
                val_metrics['val_acc_0'] = avg_val_acc
        val_metrics['val_acc_avg'] = np.mean([val_metrics['val_acc_{}'.format(x)] for x in range(len(self.dev_data))])
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
        return avg_val_loss.item()

    def configure_optimizers(self):
        return get_optimizer(self.optimizer_type, self.model, self.rate_net)

    def train_dataloader(self):
        if self.is_distributed:
            sampler = self.train_data.batch_sampler
            sampler = DistributedBatchSamplerWrapper(sampler, self.train_sampler._batch_size, shuffle=False)
            return DataLoader(self.train_data.dataset, sampler=sampler, batch_size=self.train_sampler._batch_size, num_workers=self.train_data.num_workers)
        return self.train_data

    def val_dataloader(self):
        return self.dev_data
