import numpy as np
import os
from tqdm import tqdm
from base_utils import BASE_PATH


def get_source_for_tgt(params, thershold=98.):
    result_list = list()
    for each_line in params.readlines():
        target, source, score = each_line.strip().split('\t')
        score = 100 + float(score)
        if score > thershold:
            result_list.append((target, source, score))
    result_list = sorted(result_list, key=lambda item: item[2], reverse=True)
    return result_list


def get_source_without_target_for_deletion(t_s_s_list):
    source_set = set()
    target_set = set()
    for target, source, score in t_s_s_list:
        source_set.add(source)
        target_set.add(target)


def replace_source(t_s_s_list, source_file):
    result = list()
    for each_line in tqdm(source_file.readlines()):
        for target, source, score in t_s_s_list:
            each_line = each_line.replace(source, target)
        result.append(each_line)
        print(each_line)


if __name__ == '__main__':
    infile_align_path = os.path.join(BASE_PATH, 'cross-lingual', 'align_corpus', 'rev_params')
    source_file = os.path.join(BASE_PATH, 'cross-lingual', 'align_corpus', 'format_bahasa.txt')
    replace_source(get_source_for_tgt(open(infile_align_path, 'r')), open(source_file, 'r'))
