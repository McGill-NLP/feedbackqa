"""
Script used to apply the manual fixes to a json data file.
Manual fixes are manually annotated corrections to data.
The file containing the manual fixes should be conform to the format used in --prediction, when
adding the option --write-fix-report .
"""
import argparse
import json
import logging

logger = logging.getLogger(__name__)


def fix_annotations(input_data, input_manually_fixed_data):
    output_data = {'examples': [], 'passages': input_data['passages']}

    fixes = input_manually_fixed_data['fixes']

    for i, example in enumerate(input_data['examples']):
        if (not example['question'] == fixes[i]['question'] or
                not example['source'] == fixes[i]['source']):
            raise ValueError('cannot combine the two input files')
        example['passage_id'] = fixes[i]['fix']
        output_data['examples'].append(example)
    return output_data


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input-json", help="original json file", required=True)
    parser.add_argument("--input-manually-fixed-json", help="file with manual fixes", required=True)
    parser.add_argument("--output", help="folder where to write the output", required=True)
    parser.add_argument("--format-input-json-to",
                        help="will copy and reformat the input json to this file (useful for diff")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.input_json, 'r', encoding='utf-8') as in_stream:
        input_data = json.load(in_stream)

    if args.format_input_json_to is not None:
        with open(args.format_input_json_to, "w", encoding="utf-8") as ostream:
            json.dump(input_data, ostream, indent=4, ensure_ascii=False)

    with open(args.input_manually_fixed_json, 'r', encoding='utf-8') as in_stream:
        input_manually_fixed_data = json.load(in_stream)

    output_data = fix_annotations(input_data, input_manually_fixed_data)

    with open(args.output, "w", encoding="utf-8") as ostream:
        json.dump(output_data, ostream, indent=4, ensure_ascii=False)

    logger.info('result written to {}'.format(args.output))


if __name__ == "__main__":
    main()
