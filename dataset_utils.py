import json
import logging
import os

from tqdm import tqdm
import tensorflow as tf

from base_utils import BASE_PATH, DATASET_PATH
from base_utils import clean_dir
from glob import glob

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DOMAINS = ['taxi', 'plane', 'restaurant', 'hotel', 'wear', 'hospital', 'police', 'movie', 'attraction']
METRICS = ['action', 'slots', 'intent']
clean_dir()


def get_anno_files(domains=None, anno_path=None, tpu=False):
    if domains is None:
        domains = DOMAINS

    if anno_path is None:
        anno_path = os.path.join(DATASET_PATH, 'annotations')
    anno_data_dict = dict()

    if tpu:
        for each_domain in tqdm(domains):
            for each_file in tf.io.gfile.glob(os.path.join(anno_path, each_domain)):
                with tf.io.gfile.GFile(each_file, "r") as reader:
                    anno_data = json.loads(reader)
                for k, v in anno_data['topic'].items():
                    k = k.lower()
                    if k in domains and v:
                        anno_data_dict[k].append(anno_data)
        return domains, anno_data_dict

    if len(domains) == 1 and domains[0] == 'atomeCallbot':
        anno_data_dict[domains[0]] = list()
        for each_file in glob(os.path.join(anno_path, domains[0], '*.json')):
            anno_data_dict[domains[0]].append(json.load(open(each_file, 'r')))
        return domains, anno_data_dict
    else:
        for each_domain in tqdm(domains):
            anno_data_dict[each_domain] = list()
            anno_dir = os.path.join(anno_path, each_domain)
            if os.path.exists(anno_dir):
                for each_file in glob(os.path.join(anno_dir, '*.json')):
                    with open(each_file, 'r') as in_f:
                        anno_data = json.load(in_f)
                        for k, v in anno_data['topic'].items():
                            k = k.lower()
                            if k in domains and v:
                                anno_data_dict[k].append(anno_data)
    return domains, anno_data_dict


def get_template_info(template_name=None):
    template_info = dict()
    if template_name is None:
        logging.warning('Default template for all: {}'.format(str(DOMAINS)))
        for each_topic in DOMAINS:
            # print('generate template info for: ', each_topic)
            get_template_info(each_topic)
    else:
        template_file = os.path.join(DATASET_PATH, 'templates', template_name + 'Template.json')
        template_file_out_dir = os.path.join(DATASET_PATH, 'templates_statistics')
        if not os.path.exists(template_file_out_dir):
            os.makedirs(template_file_out_dir)
        template_file_out = os.path.join(template_file_out_dir, template_name + 'Template_statistic.json')
        if os.path.exists(template_file_out):
            return json.load(open(template_file_out, 'r')), True

        if not os.path.exists(template_file):
            logging.warning('{}, doesnt exists!'.format(template_file))
            return template_info, False

        template_data = json.load(open(template_file, 'r'))

        template_info['templateName'] = template_data['templateName']
        template_data_keys = ['result', 'slots', 'domain', 'topic', 'action', 'intent']
        for each_template_key in template_data_keys:
            template_info[each_template_key] = list()
            for each_res_item in template_data[each_template_key]:
                template_info[each_template_key].append(each_res_item['propertyName'])

        with open(template_file_out, 'w+') as out_f:
            out_f.write(json.dumps(template_info))
        return template_info, True


def get_dataset_config():
    config_path = os.path.join(DATASET_PATH, 'configs')
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    templates_statistics_path = os.path.join(DATASET_PATH, 'templates_statistics')
    if not os.path.exists(templates_statistics_path):
        logging.warning('Get template statistics ...')
        get_template_info()

    for root, dirs, files in os.walk(templates_statistics_path):
        for each_file in files:
            with open(os.path.join(root, each_file), 'r') as in_f:
                temp = json.load(in_f)
                out_dict = dict()
                with open(os.path.join(config_path, '{}_config.json'.format(temp['topic'][0].lower())), 'w+') as out_f:
                    m_2_id = '{}_2_id'
                    id_2_m = 'id_2_{}'
                    for each_metric in METRICS:
                        out_dict[m_2_id.format(each_metric)] = dict(zip(temp[each_metric],
                                                                        range(len(temp[each_metric]))))
                        out_dict[id_2_m.format(each_metric)] = {v: k for (k, v) in
                                                                out_dict[m_2_id.format(each_metric)].items()}

                    out_f.write(json.dumps(out_dict))

    actions = set()
    for each_file in glob(os.path.join(config_path, '*_config.json')):
        actions.update(list(json.load(open(each_file, 'r'))['action_2_id'].keys()))
    action_dict = {
        'action_2_id': dict(zip(actions, range(len(actions)))),
        'id_2_actions': dict(zip(range(len(actions)), actions))
    }
    json.dump(action_dict, open(os.path.join(config_path, 'actions.json'), 'w+'))

    intents = set()
    for each_file in glob(os.path.join(config_path, '*_config.json')):
        intents.update(list(json.load(open(each_file, 'r'))['intent_2_id'].keys()))
    intents_dict = {
        'intent_2_id': dict(zip(intents, range(len(intents)))),
        'id_2_intent': dict(zip(range(len(intents)), intents))
    }
    json.dump(intents_dict, open(os.path.join(config_path, 'intents.json'), 'w+'))


def gen_dicts(metrics=None, domains=None):
    dict_path = os.path.join(DATASET_PATH, 'dictionary')
    if not os.path.exists(dict_path):
        os.makedirs(dict_path)

    if metrics is None:
        metrics = METRICS
    if domains is None:
        domains = DOMAINS

    domain_dicts = dict()
    for each_domain in domains:
        domain_dicts[each_domain] = dict()
        for each_metric in metrics:
            domain_dicts[each_domain][each_metric] = list()

    rm_slots = ['deskripsi', 'sinopsis']

    for root, dirs, files in os.walk(os.path.join(DATASET_PATH, 'annotations')):
        for file in tqdm(files):
            if file.endswith('.json') and os.path.split(root)[-1] in domains:
                with open(os.path.join(root, file), 'r') as in_f:
                    anno_data = json.load(in_f)
                for this_domain, v in anno_data['topic'].items():
                    if v:
                        for each_msg in anno_data['chatMessages']:
                            for each_metric in metrics:
                                for each_slot_type, slots_list in each_msg[each_metric].items():
                                    if each_slot_type not in rm_slots:
                                        for e_slot in slots_list:
                                            domain_dicts[str(this_domain).lower()][each_metric].\
                                                append('\"{}\",{}\n'.format(e_slot.strip(), each_slot_type))

    for each_domain in domains:
        for each_metric in metrics:
            with open(os.path.join(dict_path, '{}_{}.csv'.format(each_domain, each_metric)), 'w+') as out_f:
                for each_item in domain_dicts[each_domain][each_metric]:
                    out_f.write(each_item)


if __name__ == '__main__':
    # get_anno_files()
    get_template_info()
    get_dataset_config()
