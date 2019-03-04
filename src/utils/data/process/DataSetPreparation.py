import numpy as np
import shutil, os

def DataSetPreparation(src_path, valid_ratio):
    dir_names = os.listdir(src_path)
    head, tail = os.path.split(src_path)
    dst = '{}/{}'.format(head, 'data')
    for dir_name in dir_names:

        print('Processing:', dir_name)
        path_to_dir_name = '{}/{}'.format(src_path, dir_name)
        file_names = os.listdir(path_to_dir_name)
        nb = len(file_names)
        nb_valid = int(valid_ratio * nb)

        np.random.shuffle(file_names)
        file_names_valid = file_names[:nb_valid]
        file_names_train = file_names[nb_valid:]

        for phase, file_names_phase in zip(['train', 'valid'], [file_names_train, file_names_valid]):
            for file_name in file_names_phase:
                path_to_file_name_src = '{}/{}'.format(path_to_dir_name, file_name)
                path_to_dir_name_dst = '{}/{}/{}'.format(dst, phase, dir_name)
                try:
                    os.makedirs(path_to_dir_name_dst)
                except:
                    pass

                path_to_file_name_dst = '{}/{}'.format(path_to_dir_name_dst, file_name)
                shutil.move(path_to_file_name_src, path_to_file_name_dst)

    shutil.rmtree(src_path)