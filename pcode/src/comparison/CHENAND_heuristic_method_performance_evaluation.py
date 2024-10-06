from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
from beard import metrics
from mytookit.data_reader import DBReader
from scipy.spatial.distance import cosine
from tqdm import tqdm

from eutilities.metric import calc_metrics, metric_names
from eutilities.resultsaver import save_result
from eutilities.string_utils import jaccard_similarity, intersection

import pandas as pd
df_test = pd.read_csv('CHENAND_sub_CHEN_dataset_sampled_author_pair.csv')
print('df_test.shape: ', df_test.shape)

def cosine_similarity(v1, v2):
    if v1 is None or len(v1) == 0 or v2 is None or len(v2) == 0 or len(v1) != len(v2):
        return 0
    else:
        sim = 1 - cosine(v1, v2)
        if np.isnan(sim):
            # the reason is that in some case v1 or v2 are zero-like vector, e.g., [0, 0, 0, 0, ..., 0]
            # print('error encountering (NAN) when computing cosine similarity')
            return 0
        else:
            return sim


feature_and_metadata = []
for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    xmol_clean_author_name, xmol_author_pinyin_name, pid1, ao1, pid2, ao2, source1, source2, \
        xmol_aid1, coauthors1, organizations1, country1, provinces1, cities1, postcodes1, en_specter_embedding1, title1, journal_title1, pub_year1, \
        xmol_aid2, coauthors2, organizations2, country2, provinces2, cities2, postcodes2, en_specter_embedding2, title2, journal_title2, pub_year2, \
        train1_test0_val2 = row

    # metadata: xmol_clean_author_name, pid1, pid2, xmol_aid1, xmol_aid2, train1_test0_val2
    same_author = 1 if xmol_aid1 == xmol_aid2 else 0

    sim_coauthors_intersection = intersection(coauthors1, coauthors2)
    sim_affiliations_intersection = intersection(organizations1, organizations2)
    sim_country_intersection = intersection(country1, country2)
    sim_provinces_intersection = intersection(provinces1, provinces2)
    sim_cities_intersection = intersection(cities1, cities2)
    sim_postcode_intersection = intersection(postcodes1, postcodes2)
    sim_embd_infsent = -1
    sim_embd_sent2vec = -1
    sim_embd_specter = cosine_similarity(en_specter_embedding1, en_specter_embedding2)
    # sim_embd_medbert = cosine_similarity(zh_medbert_embedding1, zh_medbert_embedding2)
    sim_embd_medbert = -1
    sim_title_jaccard = jaccard_similarity([n for n in title1.split(' ') if len(n) > 2], [n for n in title2.split(' ') if len(n) > 2])
    sim_journal_jaccard = jaccard_similarity([n for n in journal_title1.split(' ') if len(n) > 2], [n for n in journal_title2.split(' ') if len(n) > 2])
    sim_pubyear_diff = 40 - abs(int(pub_year1) - int(pub_year2)) if 40 - abs(int(pub_year1) - int(pub_year2)) > 0 else 0

    feature_and_metadata.append(
        [xmol_clean_author_name, pid1, pid2, same_author, train1_test0_val2,  # [0-4]
         sim_coauthors_intersection,  # 0
         sim_affiliations_intersection,  # 1
         sim_country_intersection,  # 2
         sim_provinces_intersection,  # 3
         sim_cities_intersection,  # 4
         sim_postcode_intersection,  # 5
         sim_embd_infsent,  # 6
         sim_embd_sent2vec,  # 7
         sim_embd_specter,  # 8
         sim_embd_medbert,  # 9
         sim_title_jaccard,  # 10
         sim_journal_jaccard,  # 11
         sim_pubyear_diff  # 12
         ]
    )

# df_pairwise = pd.DataFrame(feature_and_metadata,
#                            columns=['name', 'pid1', 'pid2', 'same_author', 'train1_test0_val2',
#                                     'coauthors', 'affiliations', 'provinces', 'cities', 'postcodes',
#                                     'infsent', 'sen2vec', 'specter', 'medbert', 'titlsim',
#                                     'journal', 'pubyear'])

df_pairwise = pd.DataFrame(feature_and_metadata,
                           columns=['FN', 'pid1', 'pid2', 'same_author', 'train1_test0_val2',
                                    'CA', 'AF', 'NT', 'PV', 'CT', 'PC',
                                    'InferSent', 'Sent2Vec', 'SPECTER', 'MedBERT', 'TitleSim',
                                    'JT', 'PY'])

df = pd.read_csv('CH_EN_AND_dataset_block.csv')

df_train, df_val, df_test = df[df['train1_test0_val2'] == 1], df[df['train1_test0_val2'] == 2], df[df['train1_test0_val2'] == 0]
print('df_train, df_val, df_test: ', df_train.shape, df_val.shape, df_test.shape)


def make_heuristic_based_pairwise_author_judgement(author_profile1, author_profile2):
    common_ele = set(author_profile1).intersection(set(author_profile2))
    return len(common_ele)


def make_heuristic_based_author_group_judgement(author_instance_profile_2d):
    num_author_names = len(author_instance_profile_2d)
    connected_edges = []

    # Note make author-author similarity 2D matrix
    for i in range(num_author_names):
        for j in range(num_author_names):
            if i != j:
                num_common_ele = make_heuristic_based_pairwise_author_judgement(author_instance_profile_2d[i],
                                                                                author_instance_profile_2d[j])
                if num_common_ele > 0:
                    connected_edges.append([i, j])

    g = nx.Graph()
    nodes = list(range(num_author_names))
    g.add_nodes_from(nodes)
    g.add_edges_from([[s, e] for s, e in connected_edges])
    # g.add_edges_from([[e, s] for s, e in connected_edges])

    new_predictions = [0] * num_author_names
    author_group_label = 1
    # Note 使用图的连通性进行判别相同的作者
    for sub_g in nx.connected_components(g):
        sub_g = g.subgraph(sub_g)
        subgraph_nodes = sub_g.nodes()
        for k in subgraph_nodes:
            new_predictions[k] = author_group_label
        author_group_label += 1
    return new_predictions


def flatten_to_1d_list(input):
    input = [n if type(n) == list else [n] for n in input]
    input = list(chain.from_iterable(input))
    return input


def data_precision_round(arr, precision=2, pctg=True):
    return [round(x * 100 if pctg else x, precision) for x in arr]


num_author_names_in_testset = len(df_test)
eval_methods = ['FN', 'CA', 'AF', 'PV', 'CT', 'PC',
                # 'JT',
                'NT-PV-CT-PC',
                'AF-NT-PV-CT-PC',
                'CA-AF-NT-PV-CT-PC']

for eval_method in eval_methods:
    all_metrics = []
    # Note evaluation on the test set: df_test
    for block_name, sub_df in df_test.groupby(['xmol_clean_author_name']):
        ground_truths = sub_df['xmol_aid'].values
        # Note ############################################################################################
        """
        Note 无监督方法：
            基于姓名
            基于affiliation
            基于normalized_affiliation
            基于省份
            基于城市
            基于邮编
            基于affiliation+省份+城市
        """

        """
        TIP 添加有监督方法（机器学习+聚类）：
            ...
        """

        if eval_method == 'FN':
            predictions_basedon_name = make_heuristic_based_author_group_judgement(
                sub_df[['xmol_clean_author_name']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_name
        elif eval_method == 'CA':
            predictions_basedon_coauthor = make_heuristic_based_author_group_judgement(
                sub_df[['coauthors']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_coauthor
        elif eval_method == 'AF':
            predictions_basedon_normalizedaff = make_heuristic_based_author_group_judgement(
                sub_df[['organizations']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_normalizedaff
        elif eval_method == 'PV':
            predictions_basedon_province = make_heuristic_based_author_group_judgement(
                sub_df[['provinces']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_province
        elif eval_method == 'CT':
            predictions_basedon_city = make_heuristic_based_author_group_judgement(
                sub_df[['cities']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_city
        elif eval_method == 'PC':
            predictions_basedon_postcode = make_heuristic_based_author_group_judgement(
                sub_df[['postcodes']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_postcode
        elif eval_method == 'JT':
            predictions_basedon_journal = make_heuristic_based_author_group_judgement(
                sub_df[['journal_title']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_journal
        elif eval_method == 'NT-PV-CT-PC':
            predictions_basedon_provincecitypostcode = make_heuristic_based_author_group_judgement(
                sub_df[['countries', 'provinces', 'cities', 'postcodes']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_provincecitypostcode
        elif eval_method == 'AF-NT-PV-CT-PC':
            predictions_basedon_normalizedaff_provincecitypostcode = make_heuristic_based_author_group_judgement(
                sub_df[['organizations', 'countries', 'provinces', 'cities', 'postcodes']].apply(flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_normalizedaff_provincecitypostcode
        elif eval_method == 'CA-AF-NT-PV-CT-PC':
            predictions_basedon_all_metadata = make_heuristic_based_author_group_judgement(
                sub_df[['coauthors', 'organizations',
                        'countries', 'provinces', 'cities',
                        'postcodes']].apply(
                    flatten_to_1d_list, axis=1).values)
            predictions = predictions_basedon_all_metadata

        # Note #############################################################################################
        # Note test the performance of MAG author identifier
        # Note the clustering evaluation can not provide the Random baseline because it can not generate the ``labels_pred``

        metrics_pairedf = metrics.paired_precision_recall_fscore(labels_true=ground_truths, labels_pred=predictions)
        metrics_b3 = metrics.b3_precision_recall_fscore(labels_true=ground_truths, labels_pred=predictions)
        all_metrics.append([block_name, len(ground_truths)] + list(metrics_pairedf[-3:] + metrics_b3[-3:]))

    # note calculate the P R F1 metrics on PAIRWISE metrics
    ground_truths = df_pairwise['same_author'].values
    selected_columns = eval_method.split('-')

    predictions = df_pairwise[selected_columns].apply(lambda x: len([n for n in x if n != 0]) > 0, axis=1).values
    # Note eval the model
    pairwise_metrics = calc_metrics(ground_truths, predictions)
    pairwise_metrics = [pairwise_metrics[n] for n in metric_names]
    num_pairwise_instances = len(df_pairwise)

    # Note using block_name as the index row
    columns = ['Block', 'BlockSize', 'pP', 'pR', 'pF', 'bP', 'bR', 'bF']
    df_metric = pd.DataFrame(all_metrics, columns=columns)
    num_block_instances = sum(df_metric['BlockSize'].values)
    del df_metric['Block'], df_metric['BlockSize']
    # note calculate the paired-F1 and the B3-F1 score on BLOCK dataset
    block_metrics = df_metric._get_numeric_data().mean().values.tolist()

    # print(mean_metrics)
    print('\t'.join(metric_names + columns))
    metric_str = '\t'.join(['heuristics-' + eval_method, str(num_pairwise_instances) + '_' + str(num_block_instances)]
                           + [str(round(n * 100, 2)) for n in pairwise_metrics + block_metrics])
    print('num_author_names_in_testset: %d; method: %s; means metrics: %s'
          % (num_author_names_in_testset, eval_method, metric_str))

    save_result(spec='CHENAND-V2-Offline.txt', metrics=metric_str)
