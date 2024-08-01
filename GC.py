import os


def clearn_up_tmp_dir(patens=None):
    if patens is None:
        patens = ['*.dcd']
    from glob import glob
    for paten in patens:
        file_list = glob('/tmp/' + paten, recursive=False)
        for file in file_list:
            os.remove(file)
