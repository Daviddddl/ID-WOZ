import json
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xlwt

from dataset_utils import get_anno_files, get_template_info, DATASET_PATH, DOMAINS, METRICS
from base_utils import clean_dir

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', family='Times New Roman')

clean_dir()


def statistic_anno(domains=None, statistic_metrics=None):
    domains, anno_data_dict = get_anno_files(domains)
    if statistic_metrics is None:
        statistic_metrics = METRICS

    domain_statistic = dict()

    error_data_set = set()

    for each_domain, annos in anno_data_dict.items():
        if each_domain not in domains:
            logging.warning('Anno files have extra domain: {}'.format(each_domain))
            continue

        template_info, flag = get_template_info(each_domain)
        if not flag:
            continue

        if len(annos) == 0:
            logging.warning('{} doesnt have files.'.format(each_domain))
            continue

        each_domain_statistic = {'sum_anno': len(annos)}
        for each_metric in statistic_metrics:
            each_domain_statistic[each_metric] = dict()
            for each_item in template_info[each_metric]:
                each_domain_statistic[each_metric][each_item] = 0

        for each_anno in annos:
            # each annotation for the each domain
            for each_anno_sen in each_anno['chatMessages']:
                # each annotated sentence
                for each_metric in statistic_metrics:
                    for k, v in each_anno_sen[each_metric].items():
                        if k in each_domain_statistic[each_metric].keys():
                            if type(v) is bool:
                                if v:
                                    each_domain_statistic[each_metric][k] += 1
                            else:
                                # statistic slots
                                each_domain_statistic[each_metric][k] += len(v)
                        else:
                            # print('{} not in {}'.format(k, str(each_domain_statistic[each_metric].keys())))
                            error_data_set.add((each_anno['taskUnitId'], each_anno['subject']))

        domain_statistic[each_domain] = each_domain_statistic

    if len(error_data_set) > 0:
        print('Error data set: ', len(error_data_set), error_data_set)

    statistic_path = os.path.join(DATASET_PATH, 'statistic')
    if not os.path.exists(statistic_path):
        os.makedirs(statistic_path)
    with open(os.path.join(statistic_path, 'statistic_res.json'), 'w+') as out_f:
        out_f.write(json.dumps(domain_statistic))
    return domain_statistic


def statistic_4_bahasa_dataset(domain_statistic=None, metrics=None):
    if metrics is None:
        metrics = METRICS

    if domain_statistic is None:
        domain_statistic_file = os.path.join(DATASET_PATH, 'statistic', 'statistic_res.json')
        if os.path.exists(domain_statistic_file):
            with open(domain_statistic_file, 'r') as in_f:
                domain_statistic = json.load(in_f)
        else:
            domain_statistic = statistic_anno(statistic_metrics=metrics)

    sorted_res = dict()
    for each_domain in domain_statistic.keys():
        each_sorted_res = dict()
        label_vis_list = list()
        value_vis_list = list()

        for each_metric in metrics:
            label_list = list()
            num_list = list()
            each_res = sorted(domain_statistic[each_domain][each_metric].items(), key=lambda v: v[1], reverse=True)
            each_sorted_res[each_metric] = each_res
            for each_label, each_num in each_res:
                label_list.append(each_label)
                num_list.append(each_num)
            statistic_visualization(label_list, num_list, '{}_{}_{}'.
                                    format(each_domain, each_metric, domain_statistic[each_domain]['sum_anno']))

            label_vis_list.append(label_list)
            value_vis_list.append(num_list)

        if len(label_vis_list) >= 2 and len(value_vis_list) >= 2:
            double_bar(label_vis_list[0], value_vis_list[0],
                       label_vis_list[1], value_vis_list[1],
                       metrics[0], metrics[1])
        sorted_res[each_domain] = each_sorted_res

    # statistic_4_each_anno_excel()
    return sorted_res


def statistic_visualization(label_list, num_list, title=None):
    fontsize = 10
    color_list = ['#FFFFFF']

    if 'action' in title:
        color_list = ['#8F8FEF']
    if 'slots' in title:
        color_list = ['#FAA460']
    if 'intent' in title:
        color_list = ['#FFBFDF']

    plt.bar(range(len(label_list)), num_list, color=color_list, tick_label=label_list)
    plt.title(title)
    plt.xticks(rotation=90, fontsize=fontsize)
    plt.gca().yaxis.grid(True)
    plt.axis('tight')
    plt.xlim([-1, len(label_list)])
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_PATH, 'statistic', "{}.png".format(title)), dpi=400)
    plt.show()


def double_bar(label_1, value_1, label_2, value_2, metric_1, metric_2):
    assert len(label_1) == len(value_1)
    assert len(label_2) == len(value_2)
    fontsize = 10

    label_list = list()
    value_list = list()
    for i in range(min(len(label_1), len(label_2), len(value_1), len(value_2))):
        label_list.append('{}\n{}'.format(label_1[i], label_2[i]))
        value_list.append([value_1[i], value_2[i]])

    pd_graph = pd.DataFrame(value_list, index=label_list, columns=[metric_1, metric_2])
    pd_graph.plot(kind='bar', rot=0)

    plt.xticks(rotation=90, fontsize=fontsize)
    plt.show()


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def statistic_4_each_anno_excel(anno_dir=None):
    if anno_dir is None:
        anno_dir = os.path.join(DATASET_PATH, 'annotations')
    out_excel_file = xlwt.Workbook()
    sheet = out_excel_file.add_sheet('statistic_4_anno', cell_overwrite_ok=True)
    col_head = ["Domain", "actions", "slots", "subject", "taskUnitId"]
    row_num = 0
    for col_num in range(len(col_head)):
        sheet.write(row_num, col_num, col_head[col_num], set_style('Times New Roman', 220, True))

    for root, dirs, files in os.walk(anno_dir):
        for each_anno_file in files:
            if each_anno_file.endswith('.json') and root.split('/')[-1] in DOMAINS:
                row_num += 1
                with open(os.path.join(root, each_anno_file), 'r') as in_f:
                    anno_data = json.load(in_f)

                    # domain
                    if len(anno_data['topic']) == 1:
                        topic_list = list(anno_data['topic'].keys())
                        domain = topic_list[0]
                    else:
                        for k, v in anno_data['topic'].items():
                            if v:
                                domain = k

                    # action, slots
                    action_num = 0
                    slots_num = 0
                    for msg_item in anno_data['chatMessages']:
                        for k, v in msg_item['action'].items():
                            if v:
                                action_num += 1
                        for k, v in msg_item['slots'].items():
                            if len(v) > 0:
                                slots_num += 1
                row_data = [domain, action_num, slots_num, anno_data['subject'], anno_data['taskUnitId']]
                for col_num in range(len(col_head)):
                    sheet.write(row_num, col_num, row_data[col_num], set_style('Times New Roman', 220, True))

    out_excel_file.save(os.path.join(DATASET_PATH, 'statistic_4_anno.xls'))


if __name__ == '__main__':
    statistic_4_bahasa_dataset()
