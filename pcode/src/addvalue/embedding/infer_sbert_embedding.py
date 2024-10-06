import sys

sys.path.append('../../')
from model.embeder.sbert import ActionProcessor
from mytookit.data_reader import DBReader

import os
import numpy as np

from myconfig import cached_dir
import pandas as pd
df = pd.read_csv('sinomed_pubmed_citation_textual_content.csv')

df_sinomed_en = df[df['desc'] == 'sinomed-en']
df_sinomed_zh = df[(df['desc'] == 'sinomed-zh') & (df['is_in_our_labelled_dataset'] == 1)]
print(df_sinomed_en.shape, df_sinomed_zh.shape)

# # Note ######################################################################################
# # Note inferring sbert embedding for the translated sinomed english articles
# saved_model_base_path = os.path.join('/home/zhangli/mydisk-2t/repo/PuMed-AND-Production/cached', 'pairwise_and_model_sbert')
# # Note load the model
# processor = ActionProcessor(os.path.join(saved_model_base_path, 'allenai-specter/tuned_allenai-specter'), data=None)
# embedding_len = processor.model.get_sentence_embedding_dimension()
# print('embedding_len: ', embedding_len)
#
# res = processor.infer(content_list=df_sinomed_en['content'].values)
# del df_sinomed_en['content']
# df_sinomed_en['embedding'] = [','.join([str(m) for m in emb[1].astype(np.float16)]) for emb in res]
# df_sinomed_en.to_csv(os.path.join(cached_dir, 'sinomed-en-bert-embedding-len%d.tsv' % embedding_len), sep='\t', index=False,
#                      header=False, mode='a')

# Note ######################################################################################
# Note inferring BERT embedding (not SPECTER) for sinomed chinese articles
processor = ActionProcessor('trueto/medbert-base-chinese', data=None)
# processor = ActionProcessor(os.path.join(saved_model_base_path, 'allenai-specter-ebdlen-10/tuned_allenai-specter'), data=None)
embedding_len = processor.model.get_sentence_embedding_dimension()
print('embedding_len: ', embedding_len)

res = processor.infer(content_list=df_sinomed_zh['content'].values)
del df_sinomed_zh['content']
df_sinomed_zh['embedding'] = [','.join([str(m) for m in emb[1].astype(np.float16)]) for emb in res]
df_sinomed_zh.to_csv(os.path.join(cached_dir, 'sinomed-zh-bert-embedding-len%d.tsv' % embedding_len), sep='\t', index=False,
                     header=False, mode='a')
