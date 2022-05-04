"""
对于SinoMed，使用GPU每一次对聚类参数调整会生成一个全量消歧文件，这个脚本的目的是为了评估 基于规则的消歧结果 与 每个聚类参数对应的消歧结果
"""
import glob
import os
import sys
from itertools import chain

import numpy as np
import pandas as pd
from beard import metrics
from mytookit.data_reader import DBReader
from tqdm import tqdm

from eutilities.metric import calc_metrics, metric_names
from eutilities.resultsaver import save_result
from myconfig import cached_dir

args = sys.argv

aid_files = args[1:]
print('inputted aid file list:', aid_files)

if len(aid_files) == 0:
    aid_files = sorted(glob.glob(os.path.join(cached_dir, 'sinomed-and-result-20221211*')))

sql_pairwise = r'''
select xmol_clean_author_name,
     pid_ao1,
     pid_ao2,
     same_author,
     aid_heuristic1 == aid_heuristic2 as aid_heuristic
from (
       select xmol_clean_author_name, pid_ao1, pid_ao2, same_author, aid_heuristic1
       from (select xmol_clean_author_name,
                    concat(toString(new_pid1), '_', toString(author_position1)) as pid_ao1,
                    concat(toString(new_pid2), '_', toString(author_position2)) as pid_ao2,
                    xmol_aid1 == xmol_aid2                                  as same_author
                    -- Note using only test set
             from and_ds_ench.CHENAND_sub_CH_dataset_sampled_author_pair where train1_test0_val2 = 0)
                ANY
                LEFT join (select pid_ao as pid_ao1, author_id as aid_heuristic1
                            from and_ds_ench.sinomed_author_id_system_id_allocation_by_heuristics)
                           using pid_ao1)
       ANY
       LEFT join (select pid_ao as pid_ao2, author_id as aid_heuristic2
                   from and_ds_ench.sinomed_author_id_system_id_allocation_by_heuristics) using pid_ao2
;
'''

sql_block = r'''
select xmol_clean_author_name,
       arrayMap(x->x[1], groupArray([pid_ao, xmol_aid, aid_heuristic]) as tmp_arr) as pid_aos,
       arrayMap(x->x[2], tmp_arr)                                        as ground_truths,
       arrayMap(x->[x[3]], tmp_arr)                                        as combining_author_ids
from (
      select xmol_aid,
             xmol_clean_author_name,
             pid_ao,
             new_pid,
             aid_heuristic
      from (
               select xmol_aid,
                      xmol_clean_author_name,
                      concat(toString(new_pid), '_', toString(author_position)) as pid_ao,
                      new_pid
               from and_ds_ench.CH_EN_AND_dataset
                    -- Note denoting Chinese AND dataset
                    -- Note using only test set
               where CHAND > 0
                 and train1_test0_val2 = 0)
               ANY
               LEFT join (select pid_ao, author_id as aid_heuristic
                          from and_ds_ench.sinomed_author_id_system_id_allocation_by_heuristics) using pid_ao)
group by xmol_clean_author_name;
'''

df_pairwise_test = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql_pairwise, cached=False)
df_block_test = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql_block, cached=False)

pairwise_dataset_pidaos = set(list(df_pairwise_test['pid_ao1'].values.tolist() + df_pairwise_test['pid_ao2'].values.tolist()))
block_dataset_pidaos = set(list(chain.from_iterable(df_block_test['pid_aos'].values.tolist())))

aid_systems1 = ['aid_heuristic']
aid_systems2 = []
# Note read all aid files
for f in tqdm(aid_files):
    and_param_spec = 'p' + f.replace('.tsv', '').split('-')[-1]

    pairwise_pidao_aid_dict, block_pidao_aid_dict = {}, {}
    cnt = 0
    with open(f, 'r') as fr:
        for line in fr:
            cnt += 1
            # if cnt > 10000:
            #     break
            try:
                ns_id, sub_ns_id, pid, ao, class_label = line.strip().split('\t')
                pid_ao = str(pid) + '_' + str(ao)
                aid = pid_ao if int(class_label) == -1 else str(ns_id) + '_' + str(sub_ns_id) + '_' + str(class_label)
                if pid_ao in pairwise_dataset_pidaos:
                    pairwise_pidao_aid_dict[pid_ao] = aid
                if pid_ao in block_dataset_pidaos:
                    block_pidao_aid_dict[pid_ao] = aid
            except Exception as e:
                print(e)
                break
    aid_systems2.append(and_param_spec)
    #  Note associating aid for PAIRWISE dataset
    df_block_test[and_param_spec] = df_block_test['pid_aos'].apply(
        lambda x: [block_pidao_aid_dict.get(n) if n in block_pidao_aid_dict else '' for n in x])

    df_pairwise_test[and_param_spec] = df_pairwise_test[['pid_ao1', 'pid_ao2']].apply(
        # Note int(False) == 0
        lambda x: int(pairwise_pidao_aid_dict.get(x[0]) == pairwise_pidao_aid_dict.get(x[1])) if
        pairwise_pidao_aid_dict.get(x[0]) is not None and pairwise_pidao_aid_dict.get(x[1]) is not None
        else
        -1,  # Note -1 意味着ID不可获得
        axis=1)


# Note ##########################################################################################################################
# Note 合并多种作者ID变成DataFrame的一列
def combine_existing_and_our_author_ids(input):
    aids = input['combining_author_ids']
    aids = np.array(aids)

    our_aids = [input[k] for k in aid_systems2]
    our_aids = np.array(our_aids)
    if len(our_aids) > 0:
        aids = np.concatenate((aids, our_aids.transpose()), axis=1)

    return aids


df_block_test['combining_author_ids'] = df_block_test[['combining_author_ids'] + aid_systems2].apply(
    lambda x: combine_existing_and_our_author_ids(x), axis=1)
for c in aid_systems2:
    del df_block_test[c]

l = len(df_pairwise_test)
print('df_pairwise_test.shape: ', )
df_pairwise_test = df_pairwise_test[
    df_pairwise_test[aid_systems1 + aid_systems2].apply(lambda x: len([n for n in x if n == -1]), axis=1) == 0]
print('remove %d in df_pairwise_test because incomplete of author identifiers: ' % (l - len(df_pairwise_test)))

# Note ##########################################################################################################################
# Note 计算每一种aid的每一个区块的Micro-B3-F1
all_aid_systems = np.array(aid_systems1 + aid_systems2)
all_metrics = []
for index, row in tqdm(df_block_test.iterrows(), total=df_block_test.shape[0]):
    block_ns, pm_aos, ground_truths, combining_author_ids = row

    # print('block-size: %d' % len(pm_aos))
    num_id_systems = len(combining_author_ids[0])
    assert len(all_aid_systems) == num_id_systems

    # convert string aid -> length
    length_checker = np.vectorize(len)
    evaluated_author_ids_len = length_checker(combining_author_ids)
    author_instance_missing_ids_idx = set(np.nonzero(evaluated_author_ids_len == 0)[0])
    num_author_instance_in_block = combining_author_ids.shape[0]
    author_instance_full_ids_idx = [n for n in range(0, num_author_instance_in_block, 1) if
                                    n not in author_instance_missing_ids_idx]
    if len(author_instance_full_ids_idx) < 2:
        continue

    # Note remove empty prediction
    ground_truths = np.array(ground_truths)[author_instance_full_ids_idx].tolist()
    combining_author_ids = combining_author_ids[author_instance_full_ids_idx, :]
    for i in range(len(all_aid_systems)):
        id_system_name = all_aid_systems[i]
        a_id_system_pred = list(combining_author_ids[:, i])
        # note calculate the paired-F1 and the B3-F1 score
        metrics_pairedf = metrics.paired_precision_recall_fscore(labels_true=ground_truths, labels_pred=a_id_system_pred)
        metrics_b3 = metrics.b3_precision_recall_fscore(labels_true=ground_truths, labels_pred=a_id_system_pred)
        all_metrics.append(
            [id_system_name, block_ns, len(ground_truths)] + list(metrics_pairedf) + list(metrics_b3) + [ground_truths,
                                                                                                         a_id_system_pred])

# Note ##########################################################################################################################
# Note 计算每一种aid的平均 B3-F1
# Note using block_ns as the index row
columns = ['IdSys', 'Block', 'BlockSize', 'pP', 'pR', 'pF', 'bP', 'bR', 'bF', 'Truths', 'Predictions']
df = pd.DataFrame(all_metrics, columns=columns)
# df.to_csv(result_file, sep='\t')

sub_dfs = df.groupby(by=['IdSys'])
for id_sys_name, sub_df in sub_dfs:
    # note P R F1 on PAIRWISE dataset
    pairwise_predictions = df_pairwise_test[id_sys_name].values
    pairwise_ground_truths = df_pairwise_test['same_author'].values
    pairwise_dict = calc_metrics(pairwise_ground_truths, pairwise_predictions)
    metrics_prf1 = [pairwise_dict[n] for n in metric_names]
    num_pairwise_instances = len(df_pairwise_test)

    # note treating all the block as the SUPER BLOCK
    macro_ground_truths = list(chain.from_iterable(sub_df['Truths'].values.tolist()))
    macro_predictions = list(chain.from_iterable(sub_df['Predictions'].values.tolist()))
    assert len(macro_ground_truths) == len(macro_predictions)
    # Note MACRO B3 metrics on BLOCK dataset
    metrics_macro_b3 = metrics.b3_precision_recall_fscore(labels_true=macro_ground_truths, labels_pred=macro_predictions)

    num_block_instances = sum(sub_df['BlockSize'].values)
    del sub_df['Truths'], sub_df['Predictions'], sub_df['BlockSize']
    # Note MICRO B3 metrics on BLOCK dataset
    metrics_micro_b3 = sub_df._get_numeric_data().mean().values.tolist()
    num_blocks = len(sub_df)

    metric_str = '\t'.join(
        [id_sys_name, str(num_pairwise_instances) + '_' + str(num_block_instances)] + [str(round(n * 100, 2)) for n in list(
            metrics_prf1 + metrics_micro_b3 + list(metrics_macro_b3))])
    print('\t'.join(metric_names + ['pP', 'pR', 'pF', 'bP', 'bR', 'bF', 'Macro-bP', 'Macro-bR', 'Macro-bF']))
    print(metric_str)
    save_result(spec='CHAND-Online.txt', metrics=metric_str)
