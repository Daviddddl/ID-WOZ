import io
from tqdm import tqdm


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


if __name__ == '__main__':
    print(load_vectors('/Users/ddl/Downloads/cc.id.300.vec').keys())
