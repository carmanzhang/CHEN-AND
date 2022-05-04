import sys

from tqdm import tqdm

sys.path.append('../../')
from model.embeder.infersent import InferSentModel
from mytookit.data_reader import DBReader

import os
import numpy as np

from myconfig import cached_dir

# Note ######################################################################################

sql_template = r"""
select new_pid, content, desc
from and_ds_ench.sinomed_pubmed_citation_textual_content
where desc in ('sinomed-en', 'pubmed-en')
and is_in_our_labelled_dataset=1
and xxHash32(new_pid) %% %d=%d;
"""

infsent = InferSentModel(model_version=1)

num_segs = 1
for seg in tqdm(range(num_segs)):
    sql = sql_template % (num_segs, seg)
    print(sql)
    df = DBReader.tcp_model_cached_read('XXX', sql=sql, cached=False)
    # Note This outputs a numpy array with n vectors of dimension 4096.
    embds = infsent.infer_embedding(df['content'].values.tolist(), bsize=32)
    # print(embd)
    del df['content']
    df['embedding'] = [','.join([str(m) for m in emb.astype(np.float16)]) for emb in embds]
    df.to_csv(os.path.join(cached_dir, 'sinomed-pubmed-en-infsent-embedding.tsv'), sep='\t', index=False,
              header=False, mode='a')
