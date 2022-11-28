import re
from tqdm import tqdm
import json
import os
from base_utils import BASE_PATH, DATASET_PATH


def split_punctuation(sentence):
    return ' '.join(re.sub(r'([a-zA-Z])([,.!?\'\"\/\\%@#$\(\)\-])', r'\1 \2 ', sentence).split())


def preprocess(raw_file, output):
    assert raw_file != output
    with open(output, 'w+') as out_f:
        for each_line in open(raw_file, 'r').readlines():
            out_f.write(split_punctuation(each_line) + '\n')


def get_align_sens():
    total_num = 0
    with open(os.path.join(DATASET_PATH, 'parallels', 'align_sens.txt'), 'w+') as out_f:
        for root, dirs, files in os.walk(os.path.join(DATASET_PATH, 'parallels', 'ID2EN')):
            for each_f in tqdm(files):
                if each_f.endswith('.json'):
                    anno_json = json.load(open(os.path.join(root, each_f), 'r'))
                    for each_msg in anno_json['chatMessages']:
                        if 'translation' in each_msg.keys():
                            bahasa = split_punctuation(each_msg['message'].strip())
                            total_num += len(each_msg['translation'].strip().split(' '))
                            english = split_punctuation(each_msg['translation'].strip())
                            out_f.write(bahasa + ' ||| ' + english + '\n')

    print(total_num)

# ./fast_align -i ../output/align_sens.txt -d -v -o -p fwd_params >../output/fwd_align.txt 2>../output/fwd_err.txt
# ./fast_align -i ../output/align_sens.txt -r -d -v -o -p rev_params >../output/rev_align.txt 2>../output/rev_err.txt


if __name__ == '__main__':
    get_align_sens()

