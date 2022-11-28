import json
import os
from glob import glob
from base_utils import DATASET_PATH, BASE_PATH


def search_file_from_subject(subjects):
    err_files = list()
    for s in subjects:
        for each_file in glob(os.path.join(DATASET_PATH, 'annotations', 'taxi', '*.json')):
            anno = json.load(open(each_file, 'r'))
            if anno['subject'] == s:
                err_files.append(each_file)
    return err_files


def fix_taxi_key():
    for each_f in glob(os.path.join(DATASET_PATH, 'annotations', 'taxi', '*.json')):
        anno_data = open(each_f, 'r').read()
        anno_data = anno_data.replace('nomr_telpon_taksi', 'nomor_telpon_taksi')
        anno_data = anno_data.replace('inform_taksi__tipe', 'inform_taksi_tipe')
        anno_data = anno_data.replace('tipe_taksi', 'taksi_tipe')
        with open(each_f, 'w+') as out_f:
            out_f.write(anno_data)


if __name__ == '__main__':
    # search_file_from_subject('taxi_05092019_19')
    fix_taxi_key()
