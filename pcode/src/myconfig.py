import os

import torch

gpu_id = 0
device = "cuda:%d" % gpu_id
# device = 'cpu'

# Note resource config
latex_doc_base_dir = '/home/zhangli/mydisk-2t/repo/manuscripts/ongoning-works/and-dataset-chinese/src'
src_base_path = os.path.dirname(os.path.abspath(__file__))
proj_base_path = os.path.abspath(os.path.join(src_base_path, os.pardir))
# saved_result_path = os.path.join(src_base_path, 'result')
cached_dir = os.path.join(proj_base_path, 'cached')
saved_result_path = os.path.join(cached_dir, 'performance')

# print(cached_dir)
# Note model config
pretrained_model_path = proj_base_path = os.path.abspath('/home/zhangli/pre-trained-models/')
glove840b300d_path = os.path.join(pretrained_model_path, 'glove.840B/glove.840B.300d.txt')
fasttextcrawl300d2m_path = os.path.join(pretrained_model_path, 'fastText/crawl-300d-2M.vec')
infersent_based_path = os.path.join(pretrained_model_path, 'infersent')
