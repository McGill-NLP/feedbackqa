"""
Script that merges more than one embedding pickle file (usually obtained with --file-to-emb) into
a single pickle file.
"""
import argparse
import logging
import pickle

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", help="emb files to join", required=True, type=str, nargs='+')
    parser.add_argument("--passages", help="emb file with passages", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    merged_user_question = []
    merged_user_labels = []
    for input_to_merge in args.inputs:
        with open(input_to_merge, "rb") as in_stream:
            data_to_merge = pickle.load(in_stream)
            merged_user_question.extend(data_to_merge['question_embs'])
            merged_user_labels.extend(data_to_merge['question_labels'])

            logger.info('data file {} has {} examples'.format(
                input_to_merge, len(data_to_merge['question_embs'])))

    with open(args.passages, "rb") as in_stream:
        data_passages = pickle.load(in_stream)

    to_pickle = {'passage_header_embs': data_passages['passage_header_embs'],
                 'question_embs': merged_user_question,
                 'question_labels': merged_user_labels}

    logger.info('final size: {} passages / {} examples'.format(
        len(to_pickle['passage_header_embs']), len(to_pickle['question_embs'])))

    with open(args.output, "wb") as out_stream:
        pickle.dump(to_pickle, out_stream)


if __name__ == "__main__":
    main()
