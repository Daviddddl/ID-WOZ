import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_PATH, 'datasets', 'MULTIWOZ_BAHASA')


def clean_dir(dir_path=None):
    if dir_path is None:
        dir_path = BASE_PATH
    for root, dirs, files in os.walk(dir_path):
        for each_file in files:
            if each_file.endswith('DS_Store'):
                os.remove(os.path.join(root, each_file))
    return True


if __name__ == '__main__':
    print(DATASET_PATH)
