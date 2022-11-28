import csv
import json
import os
import re
import string

from tqdm import tqdm

from dataset_utils import DATASET_PATH, DOMAINS, METRICS, get_intent_from_anno

DICT_PATH = os.path.join(DATASET_PATH, 'dictionary')
c_marks = ['&', '>', ')', '+', ',', '"', '-', '=', '!', '@', ':', '%', '(',
           '\\', '_', ']', '/', ';', '$', "'", '?', '[', '.']
re_pattern = r"@| |\?|-|=|>|/|\[|\]|\+|,|&|\)|\(|\"|\'|!|;|\$|%|\\|_"


def get_alphas():
    alphas = set()
    for each_domain in DOMAINS:
        for root, dirs, files in os.walk(os.path.join(DATASET_PATH, 'annotations', each_domain)):
            for each_file in tqdm(files):
                for each_msg in json.load(open(os.path.join(root, each_file), 'r'))['chatMessages']:
                    for each_c in each_msg['message']:
                        if each_c in string.punctuation:
                            alphas.add(each_c)
    return list(alphas)


def gen_cooked_corpus(domains=None, metrics=None):
    if domains is None:
        domains = DOMAINS
    if metrics is None:
        metrics = METRICS
    cooked_corpus_path = os.path.join(DATASET_PATH, 'cooked_corpus')
    if not os.path.exists(cooked_corpus_path):
        os.makedirs(cooked_corpus_path)

    for each_domain in tqdm(domains):
        for each_metric in metrics:
            corpus_path = os.path.join(cooked_corpus_path, '{}_{}.txt'.format(each_domain, each_metric))
            cooked_corpus_list = list()
            dicts = csv.reader(open(os.path.join(DICT_PATH, '{}_{}.csv'.format(each_domain, each_metric)), 'r'))

            for root, dirs, files in os.walk(os.path.join(DATASET_PATH, 'annotations', each_domain)):
                for each_file in files:
                    anno_data = json.load(open(os.path.join(root, each_file), 'r'))
                    for each_msg in anno_data['chatMessages']:
                        ners = dict()
                        for k, v in each_msg[each_metric].items():
                            if len(v) > 0:
                                for each_v in v:
                                    for idx, each_token in enumerate(re.split(re_pattern, each_v)):
                                        if idx == 0:
                                            ners[each_token] = 'B-{}'.format(k.strip())
                                        else:
                                            ners[each_token] = 'I-{}'.format(k.strip())
                        for each_word in each_msg['message'].strip().split(' '):
                            each_word = each_word.strip()
                            if not each_word.isspace() and len(each_word) > 0:
                                have_end_mark = False
                                have_start_mark = False
                                start_mark = None
                                end_mark = None
                                if str(each_word)[-1] in c_marks:
                                    have_end_mark = True
                                if str(each_word)[0] in c_marks and len(each_word) > 1:
                                    have_start_mark = True

                                if have_end_mark:
                                    end_mark = each_word[-1]
                                    each_word = each_word[:-1]
                                if have_start_mark:
                                    start_mark = each_word[0]
                                    each_word = each_word[1:]

                                # get intent
                                if each_word in ners.keys():
                                    cooked_corpus_list.append((each_word, ners[each_word.strip()],
                                                               get_intent_from_anno(each_msg['intent'])))
                                else:
                                    cooked_corpus_list.append((each_word, 'O',
                                                               get_intent_from_anno(each_msg['intent'])))

                                if not end_mark is None:
                                    cooked_corpus_list.append((end_mark, 'O',
                                                               get_intent_from_anno(each_msg['intent'])))

                                if not start_mark is None:
                                    cooked_corpus_list.append((start_mark, 'O',
                                                               get_intent_from_anno(each_msg['intent'])))

                        cooked_corpus_list.append((None, None, None))
            with open(corpus_path, 'w+') as out_f:
                for each_cooked_corpus in cooked_corpus_list:
                    if each_cooked_corpus == (None, None, None):
                        out_f.write('\n')
                    else:
                        if each_cooked_corpus[0] is not None:
                            if not each_cooked_corpus[0].isspace():
                                out_f.write('{} {} {}\n'.format(each_cooked_corpus[0],
                                                                each_cooked_corpus[1],
                                                                each_cooked_corpus[2]))

            clean_empty_lines(corpus_path, rm_intent=True)
            dataset_splits(corpus_path)


def clean_empty_lines(corpus_path, rm_intent=False):
    with open(corpus_path, 'r') as in_f:
        corpus_data_list = in_f.readlines()
    res_corpus = list()
    for each_line in corpus_data_list:
        if not each_line.strip().startswith(('B-', 'I-', 'O ')):
            if rm_intent:
                each_line_splits = each_line.split(' ')
                if len(each_line_splits) == 3:
                    each_line = '{} {}\n'.format(each_line_splits[0].strip(), each_line_splits[1].strip())
            res_corpus.append(each_line)
    with open(corpus_path, 'w+') as out_f:
        for each_res_line in res_corpus:
            out_f.write(each_res_line)


def dataset_splits(corpus_path):
    # 8 1 1
    path_split = os.path.split(corpus_path)
    domain = path_split[-1].split('_')[0]
    datasets_path = os.path.join(path_split[0], domain)
    if not os.path.exists(datasets_path):
        os.makedirs(datasets_path)

    with open(corpus_path, 'r') as in_f:
        corpus = in_f.read()

    trainset = open(os.path.join(datasets_path, '{}.train'.format(domain)), 'w+')
    evalset = open(os.path.join(datasets_path, '{}.dev'.format(domain)), 'w+')
    testset = open(os.path.join(datasets_path, '{}.test'.format(domain)), 'w+')

    train_num = 0
    eval_num = 0
    test_num = 0

    corpus_ins = corpus.split('\n\n')
    for idx, each_ins in enumerate(corpus_ins):
        if idx % 10 == 0:
            testset.write(each_ins)
            testset.write('\n\n')
            test_num += 1
        elif (idx + 1) % 10 == 0:
            evalset.write(each_ins)
            evalset.write('\n\n')
            eval_num += 1
        else:
            trainset.write(each_ins)
            trainset.write('\n\n')
            train_num += 1

    trainset.close()
    testset.close()
    evalset.close()

    print(domain, train_num, eval_num, test_num)


def get_configs():
    configs_path = os.path.join(DATASET_PATH, 'configs')
    for root, dirs, files in os.walk(configs_path):
        for file in files:
            if file.endswith('.json'):
                cfig_data = json.load(open(os.path.join(root, file), 'r'))
                domain = file.split('_')[0]
                for metric, vs in cfig_data.items():
                    if not os.path.exists('txts/{}'.format(domain)):
                        os.makedirs('txts/{}'.format(domain))
                    with open('txts/{}/{}.txt'.format(domain, metric), 'w+') as out_f:
                        for k, v in vs.items():
                            out_f.write("{}:{}\n".format(k, v))

    modify_slots()


def modify_slots():
    slots_paths = os.path.join(DATASET_PATH, 'configs', 'txts')
    for root, dirs, files in os.walk(slots_paths):
        for each_file in files:
            if 'slots_2_id' in each_file:
                lines = open(os.path.join(root, each_file), 'r').readlines()
                slots = list()
                for each_line in lines:
                    slots.append(''.join(re.split(r'\d|:', each_line.strip())))
                slots_tags = list()
                slots_tags.append('O')
                for each_s in slots:
                    for t in ['B-', 'I-']:
                        slots_tags.append(t + each_s)

                slots_2_id = open(os.path.join(root, 't_slots_2_id.txt'), 'w+')
                id_2_slots = open(os.path.join(root, 't_id_2_slots.txt'), 'w+')
                for i, k in enumerate(slots_tags):
                    slots_2_id.write('{}:{}\n'.format(k, i))
                    id_2_slots.write('{}:{}\n'.format(i, k))
                slots_2_id.close()
                id_2_slots.close()


if __name__ == '__main__':
    # gen_cooked_corpus(metrics=['slots'])
    modify_slots()
