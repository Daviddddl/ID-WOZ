import os
import json
import tensorflow as tf

from tqdm import tqdm
from glob import glob
from base_utils import DATASET_PATH
from dataset_utils import get_anno_files, DOMAINS
from extract_feat_from_bert import get_concat_feats_4_1_sen

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_features_labels(lang='bahasa', anno_path=None, features_dir=None, model_path=None, tpu=False, get_feat=True):
    if features_dir is None:
        features_dir = os.path.join(DATASET_PATH, 'bert_features', lang)
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
    domains, anno_data_dict = get_anno_files(anno_path=anno_path, tpu=tpu)
    action_config = json.load(open(os.path.join(DATASET_PATH, 'configs', 'actions.json'), 'r'))
    for each_domain in domains:
        config = json.load(open(os.path.join(DATASET_PATH, 'configs', '{}_config.json'.format(each_domain)), 'r'))
        results = list()
        for each_anno in tqdm(anno_data_dict[each_domain]):
            for each_msg in each_anno['chatMessages']:
                sentence = each_msg['message']
                each_ins = {
                    'message': sentence,
                    'actions': list(),
                    'slots': dict(),
                    'intents': list(),
                    'features': list(),
                    'tokens': list()
                }
                if get_feat:
                    feats, tokens = get_concat_feats_4_1_sen(sentence, model_path=model_path)
                    each_ins['features'] = feats
                    each_ins['tokens'] = tokens

                for each_act, v in each_msg['action'].items():
                    if v:
                        each_ins['actions'].append(action_config['action_2_id'][each_act])
                if len(each_ins['actions']) == 0:
                    each_ins['actions'].append(action_config['action_2_id']['unknown'])

                for each_slot, v in each_msg['slots'].items():
                    if len(v) > 0:
                        each_ins['slots'][each_slot] = v

                intents = dict()
                for each_intent, v in each_msg['intent'].items():
                    if v is not False:
                        if v['value']:
                            intents[v['index']] = each_intent
                for each_intent in sorted(intents.items(), key=lambda k: k[0]):
                    each_ins['intents'].append(config['intent_2_id'][each_intent[1]])

                results.append(each_ins)

        # print('Finish extracting features for {}, length is {}'.format(each_domain, len(results)))
        if not tpu:
            json.dump(results, open(os.path.join(features_dir, '{}_features.json'.format(each_domain)), 'w+'))
        else:
            json.dump(results, tf.io.gfile.GFile(
                os.path.join(features_dir, '{}_features.json'.format(each_domain)), 'w+'))


def prepare_on_tpu():
    lang = 'bahasa'
    model_path = 'gs://bert-bahasa/output_test/model.ckpt'
    feature_dir = 'gs://bert-bahasa/bert_features'
    anno_path = 'gs://bert-bahasa/annotations'
    get_features_labels(lang, anno_path, feature_dir, model_path, tpu=True)


def get_all(domain=None, output=None, phrase=None):
    if output is None:
        output = os.path.join(DATASET_PATH, 'concats')
    if not os.path.exists(output):
        os.makedirs(output)

    if phrase is None:
        phrase = ['message', 'actions', 'slots', 'intents', 'tokens']

    if domain is None:
        for each_domain in DOMAINS:
            get_all(each_domain, output, phrase)
    else:
        # get_features_labels(get_feat=False)
        for each_phrase in tqdm(phrase):
            concat_path = os.path.join(output, 'concat_{}_{}.txt'.format(domain, each_phrase))
            with open(concat_path, 'w+') as out_f:
                for each_instance in json.load(open(
                        os.path.join(DATASET_PATH, 'bert_features', 'bahasa', '{}_features.json'.format(domain)))):
                    if each_phrase is 'message':
                        out_f.write(each_instance[each_phrase].replace('\n', ' ') + '\n')
                    else:
                        if len(each_instance[each_phrase]) == 0:
                            out_f.write('-1\n')
                        else:
                            out_f.write(' '.join('%s' % v for v in each_instance[each_phrase]) + '\n')


def get_tsv(phrase=None, gen_split=True):
    if phrase is None:
        phrase = ['actions', 'message']
    assert set(phrase).issubset({'actions', 'intents', 'message', 'slots'})
    content_dict = dict()
    for domain in DOMAINS:
        for p in phrase:
            concat_file = os.path.join(DATASET_PATH, 'concats', 'concat_{}_{}.txt'.format(domain, p))
            assert os.path.exists(concat_file)
            content_dict[p] = list(map(str.strip, open(concat_file, 'r').readlines()))

        output_dir = os.path.join(DATASET_PATH, 'tsvs')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, '{}_{}_{}_all.tsv'.format(domain, phrase[0], phrase[1])), 'w+') as out_f:
            for (k, v) in zip(content_dict[phrase[0]], content_dict[phrase[1]]):
                out_f.write('{}\t{}\n'.format(k.split(' ')[0], v))

    if gen_split:
        # 7 2 1
        train = open(os.path.join(DATASET_PATH, 'tsvs', 'actions_train.tsv'), 'w+')
        dev = open(os.path.join(DATASET_PATH, 'tsvs', 'actions_dev.tsv'), 'w+')
        test = open(os.path.join(DATASET_PATH, 'tsvs', 'actions_test.tsv'), 'w+')
        for each_tsv in glob(os.path.join(DATASET_PATH, 'tsvs', '*_all.tsv')):
            for i, c in enumerate(open(each_tsv, 'r').readlines()):
                if i % 10 == 1:
                    test.write(c)
                elif i % 10 in [2, 3]:
                    dev.write(c)
                else:
                    train.write(c)
        train.close()
        dev.close()
        test.close()


if __name__ == '__main__':
    # prepare_on_tpu()
    get_features_labels(get_feat=False)
    get_all()
    get_tsv()
