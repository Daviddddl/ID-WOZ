import os

from tqdm import tqdm

from base_utils import BASE_PATH


def get_wiki_corpus():
    wiki_path = os.path.join(BASE_PATH, 'bert', 'corpus', 'idwiki.txt')
    # if not os.path.exists(wiki_path):
    with open(wiki_path, 'w+') as out_f:
        for root, dirs, files in os.walk(os.path.join(BASE_PATH, 'bert', 'corpus', 'idwiki-resources')):
            for each_file in tqdm(files):
                with open(os.path.join(root, each_file), 'r') as in_f:
                    each_corp = in_f.read()
                    out_f.write(each_corp)
                out_f.write('\n\n')


def get_overall_corpus():
    corpus_root = os.path.join(BASE_PATH, 'bert', 'corpus')
    overall_corpus_path = os.path.join(corpus_root, 'overall.txt')
    if not os.path.exists(overall_corpus_path):
        with open(overall_corpus_path, 'w+') as out_f:
            for root, dirs, files in os.walk(corpus_root):
                for each_file in tqdm(files):
                    # print('Reading: ', each_file)
                    with open(os.path.join(root, each_file), 'r') as in_f:
                        out_f.write(in_f.read())
                    out_f.write('\n')

    split_by_line_count(overall_corpus_path, 25000)

    shards_path = os.path.join(corpus_root, 'shards')
    os.makedirs(shards_path)
    os.system('mv ' + str(corpus_root) + '/overall_* ' + shards_path)

    # make dirs for create_pretraining_data output
    output_dir = os.path.join(corpus_root, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def mk_shards(lines, head, src, sub):
    [des_filename, extname] = os.path.splitext(src)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w')
    try:
        fout.writelines([head])
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()


def split_by_line_count(filename, count):
    fin = open(filename, 'r')
    try:
        head = fin.readline()
        buf = []
        sub = 1
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mk_shards(buf, head, filename, sub)
                buf = []
        if len(buf) != 0:
            sub = mk_shards(buf, head, filename, sub)
    finally:
        fin.close()


if __name__ == '__main__':
    # get_wiki_corpus()
    get_overall_corpus()
