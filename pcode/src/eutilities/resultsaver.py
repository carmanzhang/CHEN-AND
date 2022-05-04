import os
import time

from myconfig import saved_result_path


def save_result(spec: str, metrics: str, desc=None):
    with open(os.path.join(saved_result_path, spec), 'a') as fw:
        time_str = time.strftime("%m-%d-%H:%M", time.localtime())
        # fw.write(time_str + '\t' + spec + '\n')
        fw.write(metrics + '\n')
        if desc is not None:
            fw.write(desc + '\n')
        # fw.write('-' * 100 + '\n')
