import sys

from model.embeder.sbert import ActionProcessor, ModelConfig

sys.path.append('../')
import logging

# Note set the log level, otherwise the log.info will not print
lg = logging.getLogger()
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import traceback

from mytookit.data_reader import DBReader

import os
import numpy as np
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr

# from myconfig import cached_dir

gpu_id = 1
device = "cuda:%d" % gpu_id
cached_dir = '/home/zhangli/mydisk-2t/repo/PuMed-AND-Production/cached'
saved_model_base_path = os.path.join(cached_dir, 'pairwise_and_model_sbert')


def cosin_sim_score(v1, v2):
    return 1 - distance.cosine(v1, v2)


def batch_cosin_sim_score(v1, v2):
    return [cosin_sim_score(a, b) for a, b in zip(v1, v2)]


def report_correlation_metrics(pred_score, true_score):
    pearsonr_res = pearsonr(pred_score, true_score)
    spearmanr_res = spearmanr(pred_score, true_score)
    return pearsonr_res, spearmanr_res


pairwise_data_sql = r"""
select arrayStringConcat(arrayFilter(x->length(x) > 0, extractAll(lowerUTF8(concat(paper_title1, ' ', abstract1)), '\\w+')),
                         ' ')                                            as content1,
       arrayStringConcat(arrayFilter(x->length(x) > 0, extractAll(lowerUTF8(concat(paper_title2, ' ', abstract2)), '\\w+')),
                         ' ')                                            as content2,
       same_author                                                       as score,
       (xxHash32(fullname) % 100 as rand) < 50 ? 1 : (rand < 75 ? 0 : 2) as train1_test0_val2
from and_ds_pm2022.our_and_dataset_pairwise_gold_standard
-- Note downsampling the majority class
where length(content1) > 20 and length(content2) > 20
and
same_author = 1 ? xxHash32(concat(fullname, 'xxxx')) % 100 < 15 : 1;"""

# Note ######################################################################################
# Note fine-tuning phase

# >>>>> Note Step 1. load dataset <<<<<<
df = DBReader.tcp_model_cached_read('XXXXX', sql=pairwise_data_sql, cached=False)
# df = pd.read_csv(os.path.join(cached_dir, 'lagos_and_pairwise_pubmed_sampled_data_for_training_sbert.tsv'), sep='\t')
df['content1'] = df['content1'].astype(str)
df['content2'] = df['content2'].astype(str)
print(df.head())

df_train, df_test, df_val = df[df['train1_test0_val2'] == 1], df[df['train1_test0_val2'] == 0], df[df['train1_test0_val2'] == 2]
del df_train['train1_test0_val2'], df_val['train1_test0_val2'], df_test['train1_test0_val2']
print('load train/val/test data', df_train.shape, df_val.shape, df_test.shape)

# >>>>> Note Step 2. choose an action in config.py and to do it <<<<<<
# models_in_use = ['allenai-specter', '/home/zhangli/pre-trained-models/biobert-v1.1']
models_in_use = ['allenai-specter']
print('available models are: ', models_in_use)
print()
for idx, model_name_or_path in enumerate(models_in_use):
    try:
        model_name = model_name_or_path[
                     model_name_or_path.rindex('/') + 1:] if '/' in model_name_or_path else model_name_or_path
        save_model_dir = os.path.join(saved_model_base_path, model_name)
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
        print(
            'loaded the %d-th model: \"%s\", may locate in \"%s\", and fine-tuned model will saved at \"%s\" if applicable' % (
                idx + 1, model_name, model_name_or_path, save_model_dir))

        processor = ActionProcessor(model_name_or_path, [df_train, df_val, df_test])
        # processor = processor.rebuild_model(concise_vector_len=ModelConfig.concise_vector_len)

        processor.model.max_seq_length = ModelConfig.max_seq_length
        print('updated max_seq_length: ', processor.model.max_seq_length)
        res = processor.fine_tune(
            save_model_path=os.path.join(save_model_dir, 'tuned_' + model_name),
            model_config=ModelConfig,
            show_progress_bar=True).evaluate()
        print(res)
        print()
    except Exception as e:
        traceback.print_exc()
        print(e)

# # Note load the model
# processor = ActionProcessor(os.path.join(saved_model_base_path, 'allenai-specter-ebdlen-10/tuned_allenai-specter'), data=None)
# embedding_len = processor.model.get_sentence_embedding_dimension()
# print('embedding_len: ', embedding_len)
#
# # Note ######################################################################################
# # Note inferring phase
# # Note load the dataset from a DATABASE
#
# data_sql_template = r"""select pm_id,
#        concat(clean_article_title, ' ', clean_abstract) as content
# from pubmed.nft_paper_20220101_parsed_fields
# where xxHash32(pm_id) %% 10 = %d;"""
#
# for seg in range(10):
#     sql = data_sql_template % seg
#     print(sql)
#     df = DBReader.tcp_model_cached_read('XXX', sql=sql, cached=False)
#
#     res = processor.infer(content_list=df['content'].values)
#     del df['content']
#     df['embedding'] = [','.join([str(m) for m in emb[1].astype(np.float16)]) for emb in res]
#     df.to_csv(os.path.join(cached_dir, 'pubmed-paper-bert-embedding-len%d.tsv' % embedding_len), sep='\t', index=False,
#               header=False, mode='a')
