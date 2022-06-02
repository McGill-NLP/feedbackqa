import logging

import torch
import torch.nn as nn
import tqdm
from torch.utils.checkpoint import checkpoint
from parlai.agents.transformer.modules import MultiHeadAttention, BasicAttention
# from parlai.agents.transformer.polyencoder import PolyBasicAttention

from feedbackQA.models.bert_encoder import get_ffw_layers
from feedbackQA.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)


class Retriever(nn.Module):

    """
    A retriever is a combination of two encoder (may be - or not - based on the same encoder),
    and it is used to generate the score between a question and a candidate.
    """

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug, topk=1):
        super(Retriever, self).__init__()
        self.bert_question_encoder = bert_question_encoder
        self.bert_paragraph_encoder = bert_paragraph_encoder
        self.tokenizer = tokenizer
        self.debug = debug
        self.max_question_len = max_question_len
        self.max_paragraph_len = max_paragraph_len
        self.softmax = torch.nn.Softmax(dim=1)
        # used for a funny bug/feature of the gradient checkpoint..
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.topk = topk

    def forward(self, **kwargs):
        """
        forward method - it will return a score if self.returns_embeddings = False,
                         two embeddings (question/answer) if self.returns_embeddings = True
        """
        raise ValueError('not implemented - use a subclass.')

    def compute_score(self, **kwargs):
        """
        returns a similarity score.
        """
        raise ValueError('not implemented - use a subclass.')

    def compute_embeddings(self, question, passages):

        if 'tt' in question:
            h_question = self.bert_question_encoder(
                question['ids'],
                question['am'],
                question['tt'])
        else:
            h_question = self.bert_question_encoder(
                question['ids'],
                question['am'])
        
        if len(passages['ids'].size()) == 2:
            if 'tt' in passages:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages['ids'],
                    passages['am'],
                    passages['tt'])
            else:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages['ids'],
                    passages['am'])

        if len(passages['ids'].size()) == 3:
            csize = passages['ids'].size()
            passages_reshaped = {}
            passages_reshaped['ids'] = passages['ids'].view(csize[0] * csize[1], csize[2])
            passages_reshaped['am'] = passages['am'].view(csize[0] * csize[1], csize[2])
            if 'tt' in passages:
                passages_reshaped['tt'] = passages['tt'].view(csize[0] * csize[1], csize[2])

            if 'tt' in passages_reshaped:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages_reshaped['ids'],
                    passages_reshaped['am'],
                    passages_reshaped['tt'])
            else:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages_reshaped['ids'],
                    passages_reshaped['am'])

            h_paragraphs_batch = h_paragraphs_batch.view(
                csize[0], csize[1], -1
            )

        return h_question, h_paragraphs_batch

    def embed_paragraph(self, paragraph):
        self.eval()
        with torch.no_grad():
            paragraph_inputs = self.tokenizer.encode_plus(
                paragraph, truncation=True, add_special_tokens=True, max_length=self.max_paragraph_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in paragraph_inputs.items()}

            paragraph_embedding = self.bert_paragraph_encoder(**inputs)
        return paragraph_embedding

    def embed_tokenized_paragraphs(self, paragraph):
        self.eval()
        with torch.no_grad():
            n_cands = paragraph['am'].size(0)
            n_cand_batchsize = 256
            passages = []
            for i in range(0, n_cands, n_cand_batchsize):
                temp_paragraph = {'ids': paragraph['ids'][i:i+n_cand_batchsize],
                    'am': paragraph['am'][i:i+n_cand_batchsize]}
                if 'tt' in paragraph:
                    temp_paragraph['tt'] = paragraph['tt'][i:i+n_cand_batchsize]
                    temp_paragraph_embedding = self.bert_paragraph_encoder(
                        temp_paragraph['ids'],
                        temp_paragraph['am'],
                        temp_paragraph['tt'])
                else:
                    temp_paragraph_embedding = self.bert_paragraph_encoder(
                        temp_paragraph['ids'],
                        temp_paragraph['am'])
                passages.append(temp_paragraph_embedding)

            paragraph_embedding = torch.cat(passages)
        return paragraph_embedding

    def embed_tokenized_question(self, question):
        self.eval()
        with torch.no_grad():
            if 'tt' in question:
                question_embedding = self.bert_question_encoder(
                    question['ids'],
                    question['am'],
                    question['tt'])
            else:
                question_embedding = self.bert_question_encoder(
                    question['ids'],
                    question['am'])

        return question_embedding

    def embed_question(self, question):
        self.eval()
        with torch.no_grad():
            question_inputs = self.tokenizer.encode_plus(
                question, truncation=True, add_special_tokens=True, max_length=self.max_question_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in question_inputs.items()}

            question_embedding = self.bert_question_encoder(**inputs)
        return question_embedding

    def predict(self, question, passages, passages_already_embedded=False,
                question_already_embedded=False, **unused_args):
        """

        :param question: str - a question (to encode)
        :param passages: list[str]: a list of passages
        :return: the prediction (index) and the normalized score.
        """
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch
            if passages_already_embedded:
                p_embs = passages
            else:
                p_embs = self.embed_paragrphs(passages)

            if question_already_embedded:
                q_emb = question
            else:
                q_emb = self.embed_question(question)

            relevance_scores = embs_dot_product(p_embs, q_emb)
            # relevance_scores = relevance_scores.squeeze(0)  # no batch dimension
            normalized_scores = self.softmax(relevance_scores)
            prob, prediction = torch.max(normalized_scores, 1)
            return relevance_scores, prediction, prob


    def embed_paragrphs(self, passages, progressbar=False):
        p_embs = []
        pg_fun = tqdm.tqdm if progressbar else lambda x: x
        for passage in pg_fun(passages):
            p_embs.append(self.embed_paragraph(passage))
        p_embs = torch.stack(p_embs, dim=1)
        return p_embs


class EmbeddingRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, debug):
        super(EmbeddingRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True

    def forward(self, question, passages):
        return self.compute_embeddings(question, passages)

    def compute_score(self, **kwargs):
        q_emb, p_embs = self.forward(**kwargs)
        return embs_dot_product(p_embs, q_emb)


def embs_dot_product(p_embs, q_emb):
    if len(p_embs.size()) == 2:
        return q_emb.mm(p_embs.t()) 
    if len(p_embs.size()) == 3:
        return torch.bmm(q_emb.unsqueeze(1), p_embs.transpose(2, 1)).squeeze(1)
    raise ValueError('Invalid dimensions of passage embeddings')

def _add_batch_dim(tensor):
    new_tensor = {}
    for k, v in tensor.items():
        new_tensor[k] = torch.unsqueeze(v, 0)
    return new_tensor


class FeedForwardRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
                 max_paragraph_len, debug, model_hyper_params, previous_hidden_size):
        super(FeedForwardRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = False

        check_and_log_hp(['retriever_layer_sizes'], model_hyper_params)
        ffw_layers = get_ffw_layers(
            previous_hidden_size * 2, model_hyper_params['dropout'],
            model_hyper_params['retriever_layer_sizes'] + [1], False)
        self.ffw_net = nn.Sequential(*ffw_layers)

    def forward(self, input_ids_question, attention_mask_question, token_type_ids_question,
                batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
                batch_token_type_ids_paragraphs):
        q_emb, p_embs = self.compute_embeddings(
            input_ids_question, attention_mask_question, token_type_ids_question,
            batch_input_ids_paragraphs, batch_attention_mask_paragraphs,
            batch_token_type_ids_paragraphs)
        _, n_paragraph, _ = p_embs.shape
        concatenated_embs = torch.cat((q_emb.unsqueeze(1).repeat(1, n_paragraph, 1), p_embs), dim=2)
        logits = self.ffw_net(concatenated_embs)
        return logits.squeeze(dim=2)

    def compute_score(self, **kwargs):
        return self.forward(**kwargs)

class PolyEncoderRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, hyper_params, debug):
        super(PolyEncoderRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True
        self.type = hyper_params['polyencoder_type']
        self.n_codes = hyper_params['poly_n_codes']
        self.attention_type = hyper_params['poly_attention_type']
        self.attention_num_heads = hyper_params['poly_attention_num_heads']
        if self.type == 'codes':
            self.codes_attention_type = hyper_params['codes_attention_type']
            self.codes_attention_num_heads = hyper_params['codes_attention_num_heads']
        embed_dim = hyper_params['embedding_size']

        # In case it's a polyencoder with code.
        if self.type == 'codes':
            # experimentally it seems that random with size = 1 was good.
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)

            # The attention for the codes.
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(
                    self.codes_attention_num_heads, embed_dim, hyper_params['dropout']
                )
            elif self.codes_attention_type == 'sqrt':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='sqrt', get_weights=False
                )
            elif self.codes_attention_type == 'basic':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='basic', get_weights=False
                )

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(
                self.attention_num_heads, embed_dim, hyper_params['dropout']
            )
        else:
            self.attention = PolyBasicAttention(
                self.type,
                self.n_codes,
                dim=2,
                attn=self.attention_type,
                get_weights=False,
            )
    def attend(self, attention_layer, queries, keys, values, mask):
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            return attention_layer(
                queries, keys, mask_ys=mask, values=values)
        elif isinstance(attention_layer, MultiHeadAttention):
            return attention_layer(
                queries, keys, values, mask)
        else:
            raise Exception('Unrecognized type of attention')

    def forward(self, question, passages):

        bsz = question['ids'].size(0)
        if 'tt' in question:
            h_question, h_question_mask = self.bert_question_encoder(
                question['ids'],
                question['am'],
                question['tt'])
        else:
            h_question, h_question_mask = self.bert_question_encoder(
                question['ids'],
                question['am'])
        
        if len(passages['ids'].size()) == 2:
            if 'tt' in passages:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages['ids'],
                    passages['am'],
                    passages['tt'])
            else:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages['ids'],
                    passages['am'])
            h_paragraphs_batch = h_paragraphs_batch.repeat(bsz, 1, 1)

        if len(passages['ids'].size()) == 3:
            csize = passages['ids'].size()
            passages_reshaped = {}
            passages_reshaped['ids'] = passages['ids'].view(csize[0] * csize[1], csize[2])
            passages_reshaped['am'] = passages['am'].view(csize[0] * csize[1], csize[2])
            if 'tt' in passages:
                passages_reshaped['tt'] = passages['tt'].view(csize[0] * csize[1], csize[2])

            if 'tt' in passages_reshaped:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages_reshaped['ids'],
                    passages_reshaped['am'],
                    passages_reshaped['tt'])
            else:
                h_paragraphs_batch = self.bert_paragraph_encoder(
                    passages_reshaped['ids'],
                    passages_reshaped['am'])
            h_paragraphs_batch = h_paragraphs_batch.view(
                csize[0], csize[1], -1
            )

        dim = h_question.size(2)
        if self.type == 'codes':
            ctxt_rep = self.attend(
                self.code_attention,
                queries=self.codes.repeat(bsz, 1, 1),
                keys=h_question,
                values=h_question,
                mask=h_question_mask,
            )
            ctxt_rep_mask = ctxt_rep.new_ones(bsz, self.n_codes).byte()

        elif self.type == 'n_first':
            # Expand the output if it is not long enough
            if h_question.size(1) < self.n_codes:
                difference = self.n_codes - h_question.size(1)
                extra_rep = h_question.new_zeros(bsz, difference, dim)
                ctxt_rep = torch.cat([h_question, extra_rep], dim=1)
                extra_mask = h_question_mask.new_zeros(bsz, difference)
                ctxt_rep_mask = torch.cat([h_question_mask, extra_mask], dim=1)
            else:
                ctxt_rep = h_question[:, 0 : self.n_codes, :]
                ctxt_rep_mask = h_question_mask[:, 0 : self.n_codes]

        ctxt_final_rep = self.attend(
            self.attention, h_paragraphs_batch, ctxt_rep, ctxt_rep, ctxt_rep_mask
        )

        return ctxt_final_rep, h_paragraphs_batch

    def compute_score(self, **kwargs):
        ctxt_final_rep, cand_embed = self.forward(**kwargs)
        scores = torch.sum(ctxt_final_rep * cand_embed, 2)
        return scores

    def embed_question(self, question):
        self.eval()
        with torch.no_grad():
            question_inputs = self.tokenizer.encode_plus(
                question, truncation=True, add_special_tokens=True, max_length=self.max_question_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_question_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in question_inputs.items()}

            question_embedding, question_mask = self.bert_question_encoder(**inputs)
        return question_embedding, question_mask

    def embed_tokenized_question(self, question):
        self.eval()
        with torch.no_grad():
            if 'tt' in question:
                question_embedding, question_mask = self.bert_question_encoder(
                    question['ids'],
                    question['am'],
                    question['tt'])
            else:
                question_embedding, question_mask = self.bert_question_encoder(
                    question['ids'],
                    question['am'])

        return question_embedding, question_mask

    def predict(self, question, passages, passages_already_embedded=False,
                question_already_embedded=False, topk=1, **unused_args):
        """

        :param question: str - a question (to encode)
        :param passages: list[str]: a list of passages
        :return: the prediction (index) and the normalized score.
        """
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch
            if passages_already_embedded:
                p_embs = passages
            else:
                p_embs = self.embed_paragrphs(passages)

            if question_already_embedded:
                q_emb, q_mask = question
            else:
                q_emb, q_mask = self.embed_question(question)

            bsz = q_emb.size(0)
            dim = q_emb.size(2)
            p_embs = p_embs.repeat(bsz, 1, 1)
            if self.type == 'codes':
                ctxt_rep = self.attend(
                    self.code_attention,
                    queries=self.codes.repeat(bsz, 1, 1),
                    keys=q_emb,
                    values=q_emb,
                    mask=q_mask,
                )
                ctxt_rep_mask = q_mask.new_ones(bsz, self.n_codes).byte()

            elif self.type == 'n_first':
                # Expand the output if it is not long enough
                if q_emb.size(1) < self.n_codes:
                    difference = self.n_codes - q_emb.size(1)
                    extra_rep = q_emb.new_zeros(bsz, difference, dim)
                    ctxt_rep = torch.cat([q_emb, extra_rep], dim=1)
                    extra_mask = q_mask.new_zeros(bsz, difference)
                    ctxt_rep_mask = torch.cat([q_mask, extra_mask], dim=1)
                else:
                    ctxt_rep = q_emb[:, 0 : self.n_codes, :]
                    ctxt_rep_mask = q_mask[:, 0 : self.n_codes]
            
            ctxt_final_rep = self.attend(
                self.attention, p_embs, ctxt_rep, ctxt_rep, ctxt_rep_mask
            )
            scores = torch.sum(ctxt_final_rep * p_embs, 2)
            # scores = scores.squeeze(0) ## Currently only supported for batchsize = 1
            normalized_scores = self.softmax(scores)
            prob, prediction = torch.topk(normalized_scores, topk, dim=-1) # b
        return scores, prediction, prob


class InversePolyEncoderRetriever(Retriever):

    def __init__(self, bert_question_encoder, bert_paragraph_encoder, tokenizer,
                 max_question_len, max_paragraph_len, hyper_params, debug):
        super(InversePolyEncoderRetriever, self).__init__(
            bert_question_encoder, bert_paragraph_encoder, tokenizer, max_question_len,
            max_paragraph_len, debug)
        self.returns_embeddings = True
        self.type = hyper_params['polyencoder_type']
        self.n_codes = hyper_params['poly_n_codes']
        self.attention_type = hyper_params['poly_attention_type']
        self.attention_num_heads = hyper_params['poly_attention_num_heads']
        if self.type == 'codes':
            self.codes_attention_type = hyper_params['codes_attention_type']
            self.codes_attention_num_heads = hyper_params['codes_attention_num_heads']
        embed_dim = hyper_params['embedding_size']

        # In case it's a polyencoder with code.
        if self.type == 'codes':
            # experimentally it seems that random with size = 1 was good.
            codes = torch.empty(self.n_codes, embed_dim)
            codes = torch.nn.init.uniform_(codes)
            self.codes = torch.nn.Parameter(codes)

            # The attention for the codes.
            if self.codes_attention_type == 'multihead':
                self.code_attention = MultiHeadAttention(
                    self.codes_attention_num_heads, embed_dim, hyper_params['dropout']
                )
            elif self.codes_attention_type == 'sqrt':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='sqrt', get_weights=False
                )
            elif self.codes_attention_type == 'basic':
                self.code_attention = PolyBasicAttention(
                    self.type, self.n_codes, dim=2, attn='basic', get_weights=False
                )

        # The final attention (the one that takes the candidate as key)
        if self.attention_type == 'multihead':
            self.attention = MultiHeadAttention(
                self.attention_num_heads, embed_dim, hyper_params['dropout']
            )
        else:
            self.attention = PolyBasicAttention(
                self.type,
                self.n_codes,
                dim=2,
                attn=self.attention_type,
                get_weights=False,
            )
    def attend(self, attention_layer, queries, keys, values, mask):
        if keys is None:
            keys = values
        if isinstance(attention_layer, PolyBasicAttention):
            return attention_layer(
                queries, keys, mask_ys=mask, values=values)
        elif isinstance(attention_layer, MultiHeadAttention):
            return attention_layer(
                queries, keys, values, mask)
        else:
            raise Exception('Unrecognized type of attention')

    def forward(self, question, passages):

        bsz = question['ids'].size(0)
        n_cand = None

        if 'tt' in question:
            h_question = self.bert_question_encoder(
                question['ids'],
                question['am'],
                question['tt'])
        else:
            h_question = self.bert_question_encoder(
                question['ids'],
                question['am'])

        if len(passages['ids'].size()) == 2:
            if 'tt' in passages:
                h_paragraphs_batch, h_paragraphs_mask_batch = self.bert_paragraph_encoder(
                    passages['ids'],
                    passages['am'],
                    passages['tt'])
            else:
                h_paragraphs_batch, h_paragraphs_mask_batch = self.bert_paragraph_encoder(
                passages['ids'],
                passages['am'])
            
            n_cand = passages['ids'].size(0)

        if len(passages['ids'].size()) == 3:
            csize = passages['ids'].size()
            passages_reshaped = {}
            passages_reshaped['ids'] = passages['ids'].view(csize[0] * csize[1], csize[2])
            passages_reshaped['am'] = passages['am'].view(csize[0] * csize[1], csize[2])
            if 'tt' in passages:
                passages_reshaped['tt'] = passages['tt'].view(csize[0] * csize[1], csize[2])

            if 'tt' in passages_reshaped:
                h_paragraphs_batch, h_paragraphs_mask_batch = self.bert_paragraph_encoder(
                    passages_reshaped['ids'],
                    passages_reshaped['am'],
                    passages_reshaped['tt'])
            else:
                h_paragraphs_batch, h_paragraphs_mask_batch = self.bert_paragraph_encoder(
                passages_reshaped['ids'],
                passages_reshaped['am'])
            
            n_cand = csize[1]

        dim = h_question.size(1)
        length = h_paragraphs_batch.size(0)
        
        if self.type == 'codes':
            cand_rep = self.attend(
                self.code_attention,
                queries=self.codes.repeat(length, 1, 1),
                keys=h_paragraphs_batch,
                values=h_paragraphs_batch,
                mask=h_paragraphs_mask_batch,
            )
            cand_rep_mask = cand_rep.new_ones(length, self.n_codes).byte()
        elif self.type == 'n_first':
            # Expand the output if it is not long enough
            if h_paragraphs_batch.size(1) < self.n_codes:
                difference = self.n_codes - h_paragraphs_batch.size(1)
                extra_rep = h_paragraphs_batch.new_zeros(length, difference, dim)
                cand_rep = torch.cat([h_paragraphs_batch, extra_rep], dim=1)
                extra_mask = h_paragraphs_mask_batch.new_zeros(length, difference)
                cand_rep_mask = torch.cat([h_paragraphs_mask_batch, extra_mask], dim=1)
            else:
                cand_rep = h_paragraphs_batch[:, 0 : self.n_codes, :]
                cand_rep_mask = h_paragraphs_mask_batch[:, 0 : self.n_codes]

        if len(passages['ids'].size()) == 2:
            cand_rep = cand_rep.repeat(bsz, 1, 1)
            cand_rep_mask = cand_rep_mask.repeat(bsz, 1)

        h_question = h_question.repeat(1, n_cand).view(-1, dim).unsqueeze(1)
        cand_final_rep = self.attend(
            self.attention, h_question, cand_rep, cand_rep, cand_rep_mask
        )
        h_question = h_question.squeeze(1).view(bsz, n_cand, -1)
        cand_final_rep = cand_final_rep.view(bsz, n_cand, -1)

        return h_question, cand_final_rep

    def compute_score(self, **kwargs):
        h_question, cand_final_rep = self.forward(**kwargs)
        scores = torch.sum(cand_final_rep * h_question, 2)
        return scores

    def embed_paragraph(self, paragraph):
        self.eval()
        with torch.no_grad():
            paragraph_inputs = self.tokenizer.encode_plus(
                paragraph, truncation=True, add_special_tokens=True, max_length=self.max_paragraph_len,
                pad_to_max_length=True, return_tensors='pt')
            tmp_device = next(self.bert_paragraph_encoder.parameters()).device
            inputs = {k: v.to(tmp_device) for k, v in paragraph_inputs.items()}

            paragraph_embedding, paragraph_mask = self.bert_paragraph_encoder(**inputs)
        return paragraph_embedding, paragraph_mask

    def embed_tokenized_paragraphs(self, paragraph):
        self.eval()
        with torch.no_grad():
            n_cands = paragraph['am'].size(0)
            n_cand_batchsize = 64
            passages = []
            passages_mask = []
            for i in range(0, n_cands, n_cand_batchsize):
                temp_paragraph = {'ids': paragraph['ids'][i:i+n_cand_batchsize],
                    'am': paragraph['am'][i:i+n_cand_batchsize]}
                if 'tt' in paragraph:
                    temp_paragraph['tt'] = paragraph['tt'][i:i+n_cand_batchsize]
                    temp_paragraph_embedding, temp_paragraph_mask = self.bert_paragraph_encoder(
                        temp_paragraph['ids'],
                        temp_paragraph['am'],
                        temp_paragraph['tt'])
                else:
                    temp_paragraph_embedding, temp_paragraph_mask = self.bert_paragraph_encoder(
                        temp_paragraph['ids'],
                        temp_paragraph['am'])
                passages.append(temp_paragraph_embedding)
                passages_mask.append(temp_paragraph_mask)
            paragraph_embedding = torch.cat(passages)
            paragraph_mask = torch.cat(passages_mask)
        return paragraph_embedding, paragraph_mask

    def embed_paragrphs(self, passages, progressbar=False):
        p_embs, p_masks = [], []
        pg_fun = tqdm.tqdm if progressbar else lambda x: x
        for passage in pg_fun(passages):
            p_emb, p_mask = self.embed_paragraph(passage) 
            p_embs.append(p_emb)
            p_masks.append(p_mask)
        p_embs = torch.stack(p_embs, dim=1)
        p_masks = torch.stack(p_masks, dim=1)
        return p_embs, p_masks
    
    def predict(self, question, passages, passages_already_embedded=False,
                question_already_embedded=False, topk=1):
        """

        :param question: str - a question (to encode)
        :param passages: list[str]: a list of passages
        :return: the prediction (index) and the normalized score.
        """
        self.eval()
        with torch.no_grad():
            # TODO this is only a single batch
            if passages_already_embedded:
                p_embs, p_mask = passages
            else:
                p_embs, p_mask = self.embed_paragrphs(passages)

            p_embs = p_embs.squeeze(0)
            p_mask = p_mask.squeeze(0)

            if question_already_embedded:
                q_emb = question
            else:
                q_emb = self.embed_question(question)

            bsz = q_emb.size(0)
            dim = q_emb.size(1)
            length = p_embs.size(0)
            n_cand = length
            # p_embs = p_embs.repeat(bsz, 1, 1)
            if self.type == 'codes':
                cand_rep = self.attend(
                    self.code_attention,
                    queries=self.codes.repeat(length, 1, 1),
                    keys=p_embs,
                    values=p_embs,
                    mask=p_mask,
                )
                cand_rep_mask = p_mask.new_ones(length, self.n_codes).byte()

            elif self.type == 'n_first':
                # Expand the output if it is not long enough
                if p_embs.size(1) < self.n_codes:
                    difference = self.n_codes - p_embs.size(1)
                    extra_rep = p_embs.new_zeros(length, difference, dim)
                    cand_rep = torch.cat([p_embs, extra_rep], dim=1)
                    extra_mask = p_mask.new_zeros(length, difference)
                    cand_rep_mask = torch.cat([p_mask, extra_mask], dim=1)
                else:
                    cand_rep = p_embs[:, 0 : self.n_codes, :]
                    cand_rep_mask = p_mask[:, 0 : self.n_codes]
            
            cand_rep = cand_rep.repeat(bsz, 1, 1)
            cand_rep_mask = cand_rep_mask.repeat(bsz, 1)

            q_emb = q_emb.repeat(1, n_cand).view(-1, dim).unsqueeze(1)
            cand_final_rep = self.attend(
                self.attention, q_emb, cand_rep, cand_rep, cand_rep_mask
            )
            q_emb = q_emb.squeeze(1).view(bsz, n_cand, -1)
            cand_final_rep = cand_final_rep.view(bsz, n_cand, -1)
            scores = torch.sum(cand_final_rep * q_emb, 2)
            # scores = scores.squeeze(0) ## Currently only supported for batchsize = 1
            normalized_scores = self.softmax(scores)
            prob, prediction = torch.topk(normalized_scores, topk, dim=-1) # b
        return scores, prediction, prob


class PolyBasicAttention(BasicAttention):
    """
    Override basic attention to account for edge case for polyencoder.
    """

    def __init__(self, poly_type, n_codes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.poly_type = poly_type
        self.n_codes = n_codes

    def forward(self, xs, ys, mask_ys=None, values=None):
        """
        Forward pass.

        Account for accidental dimensionality reduction when num_codes is 1 and the
        polyencoder type is 'codes'
        """
        lhs_emb = super().forward(xs, ys, mask_ys=mask_ys, values=values)
        if self.poly_type == 'codes' and self.n_codes == 1 and len(lhs_emb.shape) == 2:
            lhs_emb = lhs_emb.unsqueeze(self.dim - 1)
        return lhs_emb