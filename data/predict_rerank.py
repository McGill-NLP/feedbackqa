import json
import logging
import math
import pickle

import numpy as np
from tqdm import tqdm

from feedbackQA.data.data_loader import (
    get_passages_by_source,
    _encode_passages,
    get_passage_last_header, get_question, get_passage_id, is_in_distribution, OOD_STRING,
    get_passage_content2pid, )
from feedbackQA.data.reason_dataloader import encode_sentence_pair
from collections import defaultdict

logger = logging.getLogger(__name__)
import torch

def get_batched_pairs(qa_pairs, batch_size):
    result = []
    for i in range(0, len(qa_pairs), batch_size):
        result.append(qa_pairs[i: i + batch_size])
    return result


class Predictor:

    """
    Main class to generate prediction. It consider only in-domain part.
    (so, there is no model to decide if something is in-domain or out of domain)
    """

    def __init__(self, retriever_trainee, rerank_trainee, tokenizer, rerank_tokenizer, rerank_pair_encode, max_decode_length=40, topk=5, alpha=0.5):
        self.tokenizer = tokenizer
        self.rerank_tokenizer = rerank_tokenizer
        self.retriever_trainee = retriever_trainee
        self.max_question_len = self.retriever_trainee.retriever.max_question_len
        self.tokenizer = self.retriever_trainee.retriever.tokenizer
        self.retriever = retriever_trainee.retriever
        self.reranker = rerank_trainee#.model
        self.no_candidate_warnings = 0
        self.max_length = self.retriever_trainee.retriever.max_paragraph_len
        self.topk = topk
        self.max_decode_length = max_decode_length
        self.alpha = alpha
        self.rerank_pair_encode = rerank_pair_encode

    def generate_predictions(self, json_file, predict_to, multiple_thresholds, write_fix_report):

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        source2passages, _, passage_id2index = get_passages_by_source(
            json_data
        )
        source2passages, _, _, _ = _encode_passages(
            source2passages,
            self.max_question_len,
            self.tokenizer,
            do_not_encode=True
        )

        res = self.compute_results(json_data, passage_id2index, source2passages)
        relevance_scores_list, source2pred, questions, sources, source2label, source2feedback, source2oppo\
         = res
        return source2pred, source2label, source2oppo, questions


    def compute_results(self, json_data, passage_id2index, source2passages, rerank=False):

        predictions = []
        questions = []
        sources = []
        relevance_scores_list = []
        indices_of_correct_passage = []
        source2pred = defaultdict(lambda: [])
        source2label = defaultdict(lambda: [])
        source2feedback = defaultdict(lambda: [])
        source2oppo = defaultdict(lambda: 0.)
        # first collect and embed all the candidates (so we avoid recomputing them again and again
        source2embedded_passages = {}
        for source, passages in source2passages.items():
            logger.info('encoding source {}'.format(source))
            if passages:
                embedded_passages = self.retriever.embed_paragrphs(passages,
                                                                   progressbar=True)
                source2embedded_passages[source] = embedded_passages
            else:
                source2embedded_passages[source] = None

        # then loop over the examples and produce a prediction
        for example in tqdm(json_data["examples"]):
            question = example["question"]
            questions.append(question)
            source = example["source"]
            sources.append(source)
            index_of_correct_passage = passage_id2index[example["passage_id"]]

            relevance_scores, pre_select, norm_score = self.make_single_prediction(question, source,
                                                                 source2embedded_passages)
            pre_select = pre_select.squeeze(0)
            if index_of_correct_passage in pre_select.tolist():
                candidates = [source2passages[source][id_] for id_ in pre_select.tolist()]
                question_rerank = [question] * self.topk 
                final_pred, rate_score, feedback = self.make_rerank_prediction(question_rerank, candidates)
                
                final_score = rate_score.cpu() * self.alpha + norm_score.cpu() * (1.-self.alpha)
                final_pred = final_score.argmax()
                #print(final_pred, pre_select[final_pred], pre_select[0])
                source2pred[source].append(pre_select[final_pred])
                for a, f in zip(candidates, feedback):
                    source2feedback[source].append({'q': question, 'a': a, 'feedback': f})
                source2oppo[source] += 1.
            else:
                source2pred[source].append(-2) # early give up
            passages = []
                
            relevance_scores_list.append(relevance_scores.numpy())
            
            source2label[source].append(index_of_correct_passage)

        return relevance_scores_list, source2pred, questions, sources, \
                source2label, source2feedback, source2oppo

    def prepare_rerank_input(self, questions, candidates):
        enc_inp = []
        for q, c in zip(questions, candidates):
            enc_inp.append(self.rerank_pair_encode(q, c, self.max_length, self.rerank_tokenizer))
        ids = torch.stack([x['ids'].squeeze(0) for x in enc_inp], dim=0)
        am = torch.stack([x['am'].squeeze(0) for x in enc_inp], dim=0)
        enc_inp = {'ids': ids, 'am': am}#, 'tt': tt}
        return enc_inp

    def make_rerank_prediction(self, questions, candidates):
        pair_inp = self.prepare_rerank_input(questions, candidates)
        pred, score, feedback = self.reranker.step_inference(pair_inp, self.max_decode_length)
        return pred, score, feedback

    def make_single_prediction(self, question, source, source2embedded_passages,
                               question_already_embedded=False):
        if source in source2embedded_passages and source2embedded_passages[source] is not None:
            embedded_candidates = source2embedded_passages[source]
            return self.retriever.predict(question, embedded_candidates,
                                          passages_already_embedded=True,
                                          question_already_embedded=question_already_embedded,
                                          topk=self.topk)
        else:
            self.no_candidate_warnings += 1
            logger.warning('no candidates for source {} - returning 0 by default (so far, this '
                           'happened {} times)'.format(source, self.no_candidate_warnings))
            return -2, 1.0


class PredictorWithOutlierDetector(Predictor):
    """
        Generates predictionand it include also the model used to detect outliers.
    """

    def __init__(self, retriever_trainee, outlier_detector_model):
        super(PredictorWithOutlierDetector, self).__init__(retriever_trainee)
        self.outlier_detector_model = outlier_detector_model

    def make_single_prediction(self, question, source, source2embedded_passages):
        emb_question = self.retriever.embed_question(question)
        in_domain = self.outlier_detector_model.predict(emb_question)
        in_domain = np.squeeze(in_domain)
        if in_domain == 1:  # in-domain
            return super(PredictorWithOutlierDetector, self).make_single_prediction(
                emb_question, source, source2embedded_passages, question_already_embedded=True)
        else:  # out-of-domain (-1 is the result we return for out-of-domain)
            return -1, 1.0


def make_readable(passages):
    result = []
    for passage in passages:
        new_entry = {'passage_id': passage['passage_id'], 'source': passage['source'],
                     'reference_type': passage['reference_type'],
                     'section_headers': passage['reference']['section_headers']}
        result.append(new_entry)
    return result


def generate_and_log_results(relevance_scores_list, indices_of_correct_passage, normalized_scores, predict_to,
                             predictions, questions, source2passages, sources,
                             multiple_thresholds, write_fix_report, json_data):
    with open(predict_to, "w") as out_stream:
        if write_fix_report:

            fix_json = {'passages': make_readable(json_data['passages']), 'fixes': []}
            passage_content2pid = get_passage_content2pid(json_data['passages'])
            log_results_to_file(relevance_scores_list, indices_of_correct_passage, normalized_scores, out_stream,
                                predictions, questions, source2passages, sources, fix_json,
                                passage_content2pid)
            with open(predict_to + '_fix.json', 'w', encoding="utf-8") as ostream:
                json.dump(fix_json, ostream, indent=4, ensure_ascii=False)
        else:
            log_results_to_file(relevance_scores_list, indices_of_correct_passage, normalized_scores, out_stream,
                                predictions, questions, source2passages, sources, None, None)

        out_stream.write('results:\n\n')
        if multiple_thresholds:
            for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                result_message = compute_result_at_threshold(
                    predictions, indices_of_correct_passage, normalized_scores, threshold, True
                )
                logger.info(result_message)
                out_stream.write(result_message + "\n")
        else:
            result_message = compute_result_at_threshold(
                predictions, indices_of_correct_passage, normalized_scores, 0.0, False
            )
            logger.info(result_message)
            out_stream.write(result_message + "\n")



def log_results_to_file(relevance_scores_list, indices_of_correct_passage, normalized_scores, out_stream,
                        predictions, questions, source2passages, sources, fix_json=None,
                        passage_content2pid=None):
    for i in range(len(predictions)):
        relevance_scores = relevance_scores_list[i]
        ranks = np.argsort(-1 * relevance_scores)
        question = questions[i]
        prediction = predictions[i]
        index_of_correct_passage = indices_of_correct_passage[i]
        norm_score = normalized_scores[i]
        source = sources[i]
        # out_stream.write("-------------------------\n")
        # out_stream.write("question:\n\t{}\n".format(question))
        out_stream.write('{}|{}\n'.format(index_of_correct_passage, str(list(ranks[0]))[1:-1]))

        if prediction == index_of_correct_passage and prediction == -1:
            pred_outcome = "OOD_CORRECT"
        elif prediction == index_of_correct_passage and prediction >= 0:
            pred_outcome = "ID_CORRECT"
        elif prediction != index_of_correct_passage and index_of_correct_passage == -1:
            pred_outcome = "OOD_MISCLASSIFIED_AS_ID"
        elif prediction == -1 and index_of_correct_passage >= 0:
            pred_outcome = "ID_MISCLASSIFIED_AS_OOD"
        elif (prediction >= 0 and index_of_correct_passage >= 0 and
              prediction != index_of_correct_passage):
            pred_outcome = "ID_MISCLASSIFIED_AS_ANOTHER_ID"
        else:
            raise ValueError('wrong prediction/target combination')

        prediction_content = source2passages[source][prediction] if prediction >= 0 else OOD_STRING
        # out_stream.write(
        #     "prediction: {} / norm score {:3.3}\nprediction content:"
        #     "\n\t{}\n".format(
        #         pred_outcome,
        #         norm_score,
        #         prediction_content
        #     )
        # )
        target_content = source2passages[source][
            index_of_correct_passage] if index_of_correct_passage >= 0 else OOD_STRING
        # out_stream.write(
        #     "target content:\n\t{}\n\n".format(
        #         target_content
        #     )
        # )
        if fix_json is not None:
            new_entry = {}
            new_entry['source'] = source
            new_entry['question'] = question
            target_pid = passage_content2pid[source][target_content]
            new_entry['target'] = (target_pid, target_content)
            prediction_pid = passage_content2pid[source][prediction_content]
            new_entry['prediction'] = (prediction_pid, prediction_content)
            new_entry['fix'] = target_pid
            fix_json['fixes'].append(new_entry)


def generate_embeddings(ret_trainee, input_file=None, out_file=None, json_data=None,
                        embed_passages=True):
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    elif json_data:
        pass
    else:
        raise ValueError("You should specify either the input file or the json_data")

    source2passages, pid2passage, _ = get_passages_by_source(json_data)

    question_embs = []
    question_texts = []
    labels = []
    if json_data.get("examples"):
        for example in tqdm(json_data["examples"]):
            pid = get_passage_id(example)
            passage = pid2passage[pid]
            labels.append('id' if is_in_distribution(passage) else 'ood')
            question = get_question(example)
            emb = ret_trainee.retriever.embed_question(question)
            question_embs.append(emb)
            question_texts.append(question)

    passage_header_embs = []
    ood = 0
    passage_texts = []
    if embed_passages:
        for source, passages in source2passages.items():
            logger.info('embedding passages for source {}'.format(source))
            for passage in tqdm(passages):
                if is_in_distribution(passage):
                    passage_text = get_passage_last_header(passage, return_error_for_ood=True)
                    emb = ret_trainee.retriever.embed_paragraph(
                        passage_text)
                    passage_header_embs.append(emb)
                    passage_texts.append(passage_text)
                else:
                    ood += 1

    to_serialize = {"question_embs": question_embs, "passage_header_embs": passage_header_embs,
                    "question_labels": labels, "passage_texts": passage_texts,
                    "question_texts": question_texts}
    if out_file:
        with open(out_file, "wb") as out_stream:
            pickle.dump(to_serialize, out_stream)
    logger.info(
        'generated {} question embeddings and {} passage header embeddings ({} skipped because '
        'out-of-distribution)'.format(
            len(question_embs), len(passage_header_embs), ood))

    return to_serialize

def __compute_result_simple(
    predictions, indices_of_correct_passage, normalized_scores, threshold, log_threshold
):
    count = len(indices_of_correct_passage)
    ood_count = sum([x == -1 for x in indices_of_correct_passage])
    id_count = count - ood_count
    correct = 0
    id_correct = 0
    ood_correct = 0
    ood_misclassified_as_id = 0
    id_misclassified_as_ood = 0
    id_misclassified_as_id = 0

    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            after_threshold_pred = prediction
            # id_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        else:
            after_threshold_pred = -1
            # ood_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        correct += int(after_threshold_pred == indices_of_correct_passage[i])
        if indices_of_correct_passage[i] != -1 and after_threshold_pred == -1:
            # target id - prediction ood
            id_misclassified_as_ood += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] != after_threshold_pred):
            # target id - prediction id but wrong
            id_misclassified_as_id += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] == after_threshold_pred):
            # target id - prediction id and correct
            id_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred == -1:
            # target ood - prediction ood
            ood_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred != -1:
            # target ood - prediction id
            ood_misclassified_as_id += 1
        else:
            raise ValueError()
    acc = ((correct / count) * 100) if count > 0 else math.nan
    id_acc = ((id_correct / id_count) * 100) if id_count > 0 else math.nan
    ood_acc = ((ood_correct / ood_count) * 100) if ood_count > 0 else math.nan
    threshold_msg = "threshold {:1.3f}: ".format(threshold) if log_threshold else ""

    result_message = "\n{}overall: {:3}/{}={:3.2f}% acc".format(threshold_msg, correct, count, acc)
    result_message += "\n\tin-distribution: {:3}/{}={:3.2f}% acc".format(id_correct, id_count,
                                                                         id_acc)
    result_message += "\n\t\twrong because marked ood: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_ood, id_count,
        ((id_misclassified_as_ood / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\t\tmarked id but wrong candidate: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_id, id_count,
        ((id_misclassified_as_id / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\tout-of-distribution: {:3}/{}={:3.2f}% acc".format(
        ood_correct, ood_count, ood_acc)
    result_message += "\n\t------\n\t(OOD/ID classifier): correct {:3}(OOD) " \
                      "+ {:3}(ID) / {:3} = {:3.2f}% acc".format(
                          ood_correct, id_count - id_misclassified_as_ood, count,
                          100 * ((ood_correct + (id_count - id_misclassified_as_ood)) / count))

    return result_message


def compute_result_at_threshold(
    predictions, indices_of_correct_passage, normalized_scores, threshold, log_threshold
):
    count = len(indices_of_correct_passage)
    ood_count = sum([x == -1 for x in indices_of_correct_passage])
    id_count = count - ood_count
    correct = 0
    id_correct = 0
    ood_correct = 0
    ood_misclassified_as_id = 0
    id_misclassified_as_ood = 0
    id_misclassified_as_id = 0

    for i, prediction in enumerate(predictions):
        if normalized_scores[i] >= threshold:
            after_threshold_pred = prediction
            # id_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        else:
            after_threshold_pred = -1
            # ood_correct += int(after_threshold_pred == indices_of_correct_passage[i])
        correct += int(after_threshold_pred == indices_of_correct_passage[i])
        if indices_of_correct_passage[i] != -1 and after_threshold_pred == -1:
            # target id - prediction ood
            id_misclassified_as_ood += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] != after_threshold_pred):
            # target id - prediction id but wrong
            id_misclassified_as_id += 1
        elif (indices_of_correct_passage[i] != -1 and
              indices_of_correct_passage[i] == after_threshold_pred):
            # target id - prediction id and correct
            id_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred == -1:
            # target ood - prediction ood
            ood_correct += 1
        elif indices_of_correct_passage[i] == -1 and after_threshold_pred != -1:
            # target ood - prediction id
            ood_misclassified_as_id += 1
        else:
            raise ValueError()
    acc = ((correct / count) * 100) if count > 0 else math.nan
    id_acc = ((id_correct / id_count) * 100) if id_count > 0 else math.nan
    ood_acc = ((ood_correct / ood_count) * 100) if ood_count > 0 else math.nan
    threshold_msg = "threshold {:1.3f}: ".format(threshold) if log_threshold else ""

    result_message = "\n{}overall: {:3}/{}={:3.2f}% acc".format(threshold_msg, correct, count, acc)
    result_message += "\n\tin-distribution: {:3}/{}={:3.2f}% acc".format(id_correct, id_count,
                                                                         id_acc)
    result_message += "\n\t\twrong because marked ood: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_ood, id_count,
        ((id_misclassified_as_ood / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\t\tmarked id but wrong candidate: {:3}/{}={:3.2f}% err".format(
        id_misclassified_as_id, id_count,
        ((id_misclassified_as_id / id_count) * 100) if id_count > 0 else math.nan)
    result_message += "\n\tout-of-distribution: {:3}/{}={:3.2f}% acc".format(
        ood_correct, ood_count, ood_acc)
    result_message += "\n\t------\n\t(OOD/ID classifier): correct {:3}(OOD) " \
                      "+ {:3}(ID) / {:3} = {:3.2f}% acc".format(
                          ood_correct, id_count - id_misclassified_as_ood, count,
                          100 * ((ood_correct + (id_count - id_misclassified_as_ood)) / count))

    return result_message
