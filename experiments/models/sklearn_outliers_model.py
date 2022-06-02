#!/usr/bin/env python

import argparse
import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

SKLEARN_MODEL_FILE_NAME = 'sklearn_outlier_model.pkl'


def main():
    parser = argparse.ArgumentParser('script used to train the outlier detector model.'
                                     'It takes as input the embedding already produced'
                                     'by some other embedding model (e.g., BERT).')
    parser.add_argument('--embeddings', help='numpy file with embeddings', required=True)
    parser.add_argument('--output', help='will store the model output in this folder',
                        required=True)
    parser.add_argument('--model', help='the model type', default='local_outlier_factor')
    parser.add_argument('--n-neighbour', default=4, type=int)
    parser.add_argument('--keep-ood-for-questions',
                        help='will keep ood embeddings for questions- by default, they are '
                             'filtered out',
                        action='store_true')
    parser.add_argument('--train-on-questions',
                        help='will include question embeddings in train',
                        action='store_true')
    parser.add_argument('--train-on-passage-headers',
                        help='will include passage-headers in train',
                        action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.embeddings, "rb") as in_stream:
        data = pickle.load(in_stream)

    if args.train_on_questions:
        embeddings = collect_question_embeddings(args, data)
    else:
        embeddings = []

    if args.train_on_passage_headers:
        passage_header_embs = data['passage_header_embs']
        embeddings.extend(passage_header_embs)
        logger.info('found {} passage headers embs'.format(len(passage_header_embs)))

    _ = fit_sklearn_model(embeddings,
                          args.model,
                          os.path.join(args.output, SKLEARN_MODEL_FILE_NAME),
                          n_neighbors=args.n_neighbour)


def fit_sklearn_model(embeddings, model_name, output_filename, n_neighbors=4):
    logger.info('final size of the collected embeddings: {}'.format(len(embeddings)))
    embedding_array = np.concatenate(embeddings)

    if model_name == 'local_outlier_factor':
        logger.info('using local outlier factor with n_neighbour {}'.format(n_neighbors))
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.1)
    elif model_name == 'isolation_forest':
        clf = IsolationForest(contamination=0.1)
    elif model_name == 'svm':
        clf = OneClassSVM(kernel='linear')
    else:
        raise ValueError('model {} not supported'.format(model_name))
    clf.fit(embedding_array)

    logger.info('Saving OOD model to {}'.format(output_filename))
    with open(output_filename, "wb") as out_stream:
        pickle.dump(clf, out_stream)

    return clf


def collect_question_embeddings(args, data):
    question_embeddings = data['question_embs']
    question_labels = data['question_labels']
    logger.info('found {} question embeddings'.format(len(question_embeddings)))
    if not args.keep_ood_for_questions:
        embeddings = []
        for i, embedding in enumerate(question_embeddings):
            if question_labels[i] == 'id':
                embeddings.append(embedding)
        logger.info('\t{} question embeddings remain after filtering'.format(len(embeddings)))
    else:
        embeddings = question_embeddings
    return embeddings


if __name__ == "__main__":
    main()
