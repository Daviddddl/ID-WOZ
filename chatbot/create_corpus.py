import yaml
import os
import json
from tqdm import tqdm
from base_utils import DATASET_PATH
from dataset_utils import DOMAINS

# put the yaml to '/home/didonglin/anaconda3/envs/tensorflow/lib/python3.6/site-packages/chatterbot_corpus/data'

def gen_corpus():
    for domain in tqdm(DOMAINS):
        messages = open(os.path.join(DATASET_PATH, 'concats', 'concat_{}_message.txt'.format(domain))).readlines()

        corpus = {
            'categories': [domain],
            'conversations': list()
        }
        for couple in list(zip(messages[::2], messages[1::2])):
            corpus['conversations'].append([couple[0].strip(), couple[1].strip()])
        yaml.dump(corpus, open('{}.yml'.format(domain), 'w+'))


def get_corpus_4_collection():
    data_dir = '/Users/ddl/Documents/collection_dialogue/output'
    corpus = {
        'categories': 'collection',
        'conversations': list()
    }
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            json_data = json.load(open(os.path.join(root, f), 'r'))
            messages = json_data['manual_dictation'].split('\n')
            for couple in list(zip(messages[::2], messages[1::2])):
                corpus['conversations'].append([couple[0].strip(), couple[1].strip()])
    yaml.dump(corpus, open('collection.yml', 'w+'))


if __name__ == '__main__':
    # gen_corpus()
    get_corpus_4_collection()
