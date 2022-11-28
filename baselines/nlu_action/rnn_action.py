# coding=utf-8
import os
import json

from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry
# from base_utils import DATASET_PATH
DATASET_PATH = '/Users/ddl/PycharmProjects/multiwoz-bahasa/datasets/MULTIWOZ_BAHASA'

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

cur_domain = 'hotel'
inputs_from = 'bahasa'  # 'bahasa', 'multi'


def get_config(domain=cur_domain):
    config_path = os.path.join(DATASET_PATH, 'configs', '{}_config.json'.format(domain))
    assert os.path.exists(config_path)
    config_dict = json.load(open(config_path, 'r'))
    return config_dict['id_2_action']


def get_inputs_labels(domain=cur_domain):
    input_dict = os.path.join(DATASET_PATH, 'bert_features', '{}_tpu'.format(inputs_from),
                              'concat_{}_message_output.json'.format(domain))
    assert os.path.exists(input_dict)

    input_features = list()
    for each_line in open(input_dict, 'r').readlines():
        each_json = json.loads(each_line)
        current_feature = list()
        for each_feat in each_json['features']:
            current_feature.extend(each_feat['layers'][0]['values'])
        input_features.append(current_feature)

    lables = list()
    for each_line in open(os.path.join(DATASET_PATH, 'concats',
                                       'concat_{}_actions.txt'.format(domain)), 'r').readlines():
        lables.append(int(each_line.strip().split(' ')[0]))
    return input_features, lables


@registry.register_problem
class RnnAction(text_problems.Text2ClassProblem):
    ROOT_DATA_PATH = 'data_dir'
    PROBLEM_NAME = 'rnn_action'

    @property
    def is_generate_per_split(self):
        return True

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 5,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def approx_vocab_size(self):
        return 2 ** 16

    @property
    def num_classes(self):
        return len(get_config('hotel').keys())

    @property
    def vocab_filename(self):
        return self.PROBLEM_NAME + ".vocab.%d" % self.approx_vocab_size

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        inputs, labels = get_inputs_labels(cur_domain)
        for (feat, lbl) in zip(inputs, labels):
            yield {
                "inputs": feat,
                "label": int(lbl)
            }


"""
PROBLEM_NAME='rnn_action'
DATA_DIR='data_dir'
TMP_DIR='tmp_dir'
OUTPUT_DIR='outputs'
t2t-datagen --t2t_usr_dir=. --data_dir=$DATA_DIR --tmp_dir=$TMP_DIR --problem=$PROBLEM_NAME
t2t-trainer --t2t_usr_dir=. --data_dir=$DATA_DIR --problem=$PROBLEM_NAME --model=transformer --hparams_set=transformer_base --output_dir=$OUTPUT_DIR
"""
