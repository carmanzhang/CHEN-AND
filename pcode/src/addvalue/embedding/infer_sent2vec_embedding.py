import sys

from tqdm import tqdm

sys.path.append('../../')
from model.embeder.comp_gram import Sent2VecModel
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

# sent2vec_model = Sent2VecModel(model_name='WikiSentVec_wiki_unigrams')
sent2vec_model = Sent2VecModel(model_name='BioSentVec_PubMed_MIMICIII-bigram_d700')

num_segs = 1
for seg in tqdm(range(num_segs)):
    sql = sql_template % (num_segs, seg)
    print(sql)
    df = DBReader.tcp_model_cached_read('XXX', sql=sql, cached=False)
    # num_threads <=0 meaning using all threads
    embds = sent2vec_model.infer_embedding(df['content'].values.tolist(), num_threads=-1)
    del df['content']
    df['embedding'] = [','.join([str(m) for m in emb.astype(np.float16)]) for emb in embds]
    df.to_csv(os.path.join(cached_dir, 'sinomed-pubmed-en-sent2vec-embedding.tsv'), sep='\t', index=False,
              header=False, mode='a')
