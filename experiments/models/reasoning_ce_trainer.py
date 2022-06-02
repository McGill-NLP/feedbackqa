import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, DistributedSampler

from feedbackQA.models.optimizer import get_optimizer
from feedbackQA.data.data_loader import DistributedSamplerWrapper
from feedbackQA.data.generation_dataloader import encode_sentence_batch
from feedbackQA.data.multitask_dataloader import DistributedBatchSamplerWrapper
from feedbackQA.models.retriever_trainer import soft_cross_entropy, prepare_soft_targets
from feedbackQA.models.general_encoder import compute_average_with_padding

from transformers import BartTokenizerFast
import torch.nn.functional as F

import numpy as np

def trunc_model_out(feedback):
    fb_tokens = feedback.split()
    feedback = ' '.join(fb_tokens[:50])
    if not feedback.endswith('.'):
        if len(fb_tokens) < 40 and not '.' in feedback:
            feedback += '.'
        else:
            index_of_comma = [i for i, x in enumerate(feedback) if x=='.']
            feedback = feedback[:index_of_comma[-1]+1]
    return feedback

class ReasonTrainer(pl.LightningModule):

    """
    Given the q and a, generate feedback and rating
    """

    def __init__(self, model, train_data, train_sampler, dev_data, loss_type,
                 optimizer_type, train_source='batch', eval_source='inline', is_distributed=False, 
                 pad_id=-1, tokenizer=None, do_rl=False, rate_wght=1., num_class=4):
        super(ReasonTrainer, self).__init__()
        self.model = model
        self.model.to('cuda')
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
        self.do_rl = do_rl
        self.rate_wght = rate_wght
        self.rate_loss_func = nn.KLDivLoss()
        hsz = self.model.bert_question_encoder.bert.config.d_model
        self.num_class = num_class
        self.rate_net = nn.Sequential(nn.Linear(hsz, 128), nn.GELU(), nn.Linear(128, num_class))
        self.rate_net.to('cuda')

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def step_inference(self, encoder_input, max_len=40):
        pad_id = self.tokenizer.pad_token_id

        decode_outs = self.model.bert_question_encoder.bert.generate(
            input_ids=encoder_input['ids'].cuda(),
            #decoder_input_ids=decoder_input['ids'],
            num_beams=3,
            attention_mask=encoder_input['am'].cuda(),
            num_return_sequences=1,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_len,
            min_length=3,
            early_stopping=False
        )
        decoded = self.tokenizer.batch_decode(decode_outs.cpu().numpy(), skip_special_tokens=True)
        for i, dec in enumerate(decoded):
            decoded[i] = trunc_model_out(dec)
        decoded_inp, _ = encode_sentence_batch(decoded, 40, self.tokenizer)
        encoder_outputs = self.model.bert_question_encoder.bert.model.encoder(
                input_ids=encoder_input['ids'].cuda(), attention_mask=encoder_input['am'].cuda())
        assert not torch.isnan(encoder_outputs[0]).any()
        logits, decoder_state, _ = self.model.bert_question_encoder.bert(
            input_ids=None,
            decoder_input_ids=decoded_inp['ids'].cuda(),
            encoder_outputs=encoder_outputs,
            use_cache=False,
            output_hidden_states=True,
            attention_mask=encoder_input['am'].cuda()
        )
        rating_logits = self.rate_net(decoder_state[-1][:, -1, :])
        rating_score = F.softmax(rating_logits, dim=-1)
        rating_score = rating_score[:, -1]
        return rating_score.argmax(dim=-1), rating_score, decoded

    def step_inference_given_rating(self, encoder_input, target_rating, max_len=40):
        pad_id = self.tokenizer.pad_token_id
        beam_size = 3
        decode_outs = self.model.bert_question_encoder.bert.generate(
            input_ids=encoder_input['ids'].cuda(),
            #decoder_input_ids=decoder_input['ids'],
            num_beams=beam_size,
            attention_mask=encoder_input['am'].cuda(),
            num_return_sequences=beam_size,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=41,
            min_length=3,
            early_stopping=False
        )
        decoded = self.tokenizer.batch_decode(decode_outs.cpu().numpy(), skip_special_tokens=True)
        for i, dec in enumerate(decoded):

            decoded[i] = trunc_model_out(dec)
            
        decoded_inp, _ = encode_sentence_batch(decoded, 40, self.tokenizer)
        encoder_outputs = self.model.bert_question_encoder.bert.model.encoder(
                input_ids=encoder_input['ids'].repeat(beam_size, 1).cuda(), attention_mask=encoder_input['am'].repeat(beam_size, 1).cuda())
        assert not torch.isnan(encoder_outputs[0]).any()
        logits, decoder_state, _ = self.model.bert_question_encoder.bert(
            input_ids=None,
            decoder_input_ids=decoded_inp['ids'].cuda(),
            encoder_outputs=encoder_outputs,
            use_cache=False,
            output_hidden_states=True,
            attention_mask=encoder_input['am'].repeat(beam_size, 1).cuda()
        )
        logits = F.log_softmax(logits, dim=-1)
        decoded_logits = torch.gather(logits, 2, decoded_inp['ids'].cuda().unsqueeze(2))
        
        gen_score = decoded_logits.sum(dim=1).squeeze() / decoded_inp['ids'].cuda().ne(self.tokenizer.pad_token_id).float().sum(dim=1)
        
        rating_logits = self.rate_net(decoder_state[-1][:, -1, :])
        rating_score = F.log_softmax(rating_logits, dim=-1)
        rating_score = rating_score[:, target_rating]
        score = rating_score + gen_score
        best = score.argmax(dim=-1)
        decoded_best = decoded[best.item()]
        label_rating_dist = np.zeros(4)
        label_rating_dist[target_rating] = 1.
        rating_dist = 0.4 * F.softmax(rating_logits[best], dim=-1) + 0.6 * torch.tensor(label_rating_dist).cuda()
        return rating_dist, decoded_best

    def step_helper_generator_inference(self, batch):

        encoder_input = batch['pair']
        gt_feedback_text = batch['org_feedback']
        decoder_input = batch['decoder_input']
        decoder_target = batch['decoder_target']['ids']
        target_rating = batch['mean_rating']
        target_rating_acc = batch['rating']
        pad_id = self.tokenizer.pad_token_id

        decode_outs = self.model.bert_question_encoder.bert.generate(
            input_ids=encoder_input['ids'],
            #decoder_input_ids=decoder_input['ids'],
            num_beams=5,
            num_return_sequences=1,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            min_length=3,
            max_length=32
        )
        decoded = self.tokenizer.batch_decode(decode_outs.cpu().numpy(), skip_special_tokens=True)
        for i, dec in enumerate(decoded):
            decoded[i] = gt_feedback_text[i]#trunc_model_out(dec)
        bos = torch.full((decode_outs.size(0), 1), self.tokenizer.bos_token_id).cuda().long()
        decoded_inp, _ = encode_sentence_batch(decoded, 40, self.tokenizer)
        encoder_outputs = self.model.bert_question_encoder.bert.model.encoder(
                input_ids=encoder_input['ids'], attention_mask=encoder_input['am'])        
        assert not torch.isnan(encoder_outputs[0]).any()
        logits, decoder_state, _ = self.model.bert_question_encoder.bert(
            input_ids=None,
            decoder_input_ids=decoded_inp['ids'].cuda(),
            encoder_outputs=encoder_outputs,
            use_cache=False,
            output_hidden_states=True,
            attention_mask=encoder_input['am'].cuda()
        )
        rating_logits = self.rate_net(decoder_state[-1][:, -1, :])
        rating_prob = F.log_softmax(rating_logits, dim=-1)
        rating_loss = self.rate_loss_func(rating_prob, target_rating.type(torch.float16))
        rating_acc = pl.metrics.functional.accuracy(rating_logits, target_rating_acc, num_classes=4)
        rating_pred = torch.argmax(rating_logits, dim=-1)
        del encoder_outputs
        dim = logits.size(-1)
        logits = logits.contiguous().view(-1, dim)

        return rating_loss, rating_acc, rating_pred

    def step_helper_generator(self, batch):
        encoder_input = batch['pair']
        decoder_input = batch['decoder_input']
        decoder_target = batch['decoder_target']['ids']
        target_rating = batch['rating']
        target_rating_dist = batch['reason_mean_rating']

        encoder_outputs = self.model.bert_question_encoder.bert.model.encoder(
                                input_ids=encoder_input['ids'], attention_mask=encoder_input['am'])

        logits, decoder_state, encoder_state = self.model.bert_question_encoder.bert(
            input_ids=None,
            decoder_input_ids=decoder_input['ids'],
            encoder_outputs=encoder_outputs,
            use_cache=False,
            attention_mask=encoder_input['am'],
            output_hidden_states=True,
        )
        rating_logits = self.rate_net(decoder_state[-1][:, -1, :])
        rating_logits = F.log_softmax(rating_logits)
        rating_loss = F.kl_div(rating_logits, target_rating_dist.type_as(rating_logits))
        rating_acc = pl.metrics.functional.accuracy(rating_logits, target_rating, num_classes=self.num_class)
        rating_pred = torch.argmax(rating_logits, dim=-1)
        dim = logits.size(-1)
        logits = logits.contiguous().view(-1, dim)
        generation_loss = F.cross_entropy(logits, decoder_target.view(-1), ignore_index=self.pad_id)
        return rating_loss, rating_acc, generation_loss, rating_pred
    
    def training_step(self, batch, batch_idx):
        rating_loss, rating_acc, generation_loss, _ = self.step_helper_generator(batch)
        tensorboard_logs = {'rating_train_loss': rating_loss, 'generation_train_loss': generation_loss}
        return {
            'loss': self.rate_wght * rating_loss + generation_loss,
            'acc': rating_acc, 
            'rating_loss': rating_loss, 
            'generation_loss': generation_loss,
            'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataset_number=0):
        # if self.dev_data is a dataloader, there is no provided
        # dataset_number, hence the default value at 0
        #if 'decoder_target' in batch.keys():
        if self.rate_wght > 0:
            rating_val_loss, rating_val_acc, rating_pred = self.step_helper_generator_inference(batch)
            _, rating_val_acc_gold, _, _ = self.step_helper_generator(batch)
        else:
            rating_val_loss, rating_val_acc, rating_pred = self.step_helper_generator(batch)

        return {'val_loss_' + str(dataset_number): rating_val_loss, 
                'val_acc_' + str(dataset_number): rating_val_acc, 
                'val_gold_acc_' + str(dataset_number): rating_val_acc_gold,
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
                    avg_val_gold_acc = self._comput_mean_for_metric(dataset_index, 'val_gold_acc_', outputs)
                    val_metrics['val_acc_' + str(dataset_index)] = avg_val_acc
                    val_metrics['val_gold_acc_' + str(dataset_index)] = avg_val_gold_acc
                    #avg_val_f1 = self._comput_mean_for_metric(dataset_index, 'val_f1_', outputs)

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
                avg_val_acc = self._comput_mean_for_metric(None, 'val_f1_', outputs)
                #val_metrics['val_f1_0'] = avg_val_f1
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
