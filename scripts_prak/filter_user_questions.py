"""
Script used to find the overlap between mode data files - in term of passages.
Given one (or more than one file) FS containing passages and examples, and given a reference file R
containing some passages, it will extract from FS only the examples related to the passages
that area also present in R.
"""
import argparse
import json
import logging
import ntpath
import os

from bert_reranker.data.data_loader import get_passages_by_source, get_examples, get_passage_id, \
    get_passage_content2pid, get_passage_last_header

logger = logging.getLogger(__name__)


def filter_user_questions(input_to_filter, faq_contents):
    '''
    This function takes the data from the input_to_filter json file, and only
    returns the examples that align with the faq_contents.

    input_to_filer: str, filename of the json to filter
    faq_contents: set, all faq questions in a set.
    '''
    with open(input_to_filter, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    _, pid2passage, _ = get_passages_by_source(input_data, keep_ood=True)

    filtered_example = []
    examples = get_examples(input_data, keep_ood=True)
    for example in examples:
        related_pid = get_passage_id(example)
        related_passage = pid2passage[related_pid]
        if get_passage_last_header(related_passage) in faq_contents:
            filtered_example.append(example)

    logger.info(
        'file {}: passage size {} / pre-filtering example size {} / post filtering examples size'
        ' {}'.format(input_to_filter, len(input_data['passages']), len(examples),
                     len(filtered_example)))

    return {'examples': filtered_example, 'passages': input_data['passages']}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", help="join files to filter", required=True, type=str, nargs='+')
    parser.add_argument("--faq-file", help="file containing the reference FAQ", required=True)
    parser.add_argument("--output", help="folder that will contain the output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.faq_file, 'r', encoding='utf-8') as in_stream:
        faq_file = json.load(in_stream)
        faq_contents = get_faq_contents(faq_file)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for input_to_filter in args.inputs:
        to_dump = filter_user_questions(input_to_filter, faq_contents)

        file_name = ntpath.basename(input_to_filter)
        with open(os.path.join(args.output, file_name), "w", encoding="utf-8") as ostream:
            json.dump(to_dump, ostream, indent=4, ensure_ascii=False)


def get_faq_contents(faq_file):
    _, faq_pid2passage, _ = get_passages_by_source(faq_file, keep_ood=True)
    faq_passages = list(faq_pid2passage.values())
    source2faq_contents = get_passage_content2pid(faq_passages, duplicates_are_ok=True)

    all_source_contents = []
    for source_contents in source2faq_contents.values():
        all_source_contents.extend(source_contents)
    all_source_contents = set(all_source_contents)

    logger.info('passages in reference file: {}'.format(len(all_source_contents)))
    return all_source_contents


if __name__ == "__main__":
    main()
