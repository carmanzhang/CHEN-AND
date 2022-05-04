'''
using available metadata to develop disambiguation model for SinoMed+PubMed Chinese authors
'''
import os

import joblib
import numpy as np
import pandas as pd
from beard.metrics import b3_precision_recall_fscore, paired_precision_recall_fscore
from matplotlib import pyplot as plt
from mytookit.data_reader import DBReader
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils import shuffle
from tqdm import tqdm

from eutilities.metric import calc_metrics, metric_names
from eutilities.resultsaver import save_result
from eutilities.string_utils import intersection, jaccard_similarity
from model.regression import ModelName, use_regression
from myconfig import cached_dir

sql = r'''
select xmol_clean_author_name, xmol_author_pinyin_name, concat(toString(new_pid1), '_', toString(author_position1)) as pid_ao1, concat(toString(new_pid2), '_', toString(author_position2)) as pid_ao2, xmol_aid1, xmol_raw_citation1, coauthors1, author_affiliations1, author_countries1, author_provinces1, author_cities1, author_postcodes1, en_infersent_embedding1, en_sent2vec_embedding1, en_specter_embedding1, jd1, st1, title1, journal_title1, pub_year1, xmol_aid2, xmol_raw_citation2, coauthors2, author_affiliations2, author_countries2, author_provinces2, author_cities2, author_postcodes2, en_infersent_embedding2, en_sent2vec_embedding2, en_specter_embedding2, jd2, st2, title2, journal_title2, pub_year2, commonness, train1_test0_val2
from and_ds_ench.CHENAND_sub_CHEN_dataset_sampled_author_pair ANY
         inner join (select pid_ao as pid_ao1, toInt32(round(log2(block_size))) as commonness
                     from and_ds_ench.sinopubmed_ambiguous_author_with_nameblocking_en) using pid_ao1
;
'''

df = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)

df_train, df_val, df_test = df[df['train1_test0_val2'] == 1], df[df['train1_test0_val2'] == 2], df[df['train1_test0_val2'] == 0]
print('df_train, df_val, df_test: ', df_train.shape, df_val.shape, df_test.shape)


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


def data_precision_round(arr, precision=2, pctg=True):
    return [round(x * 100 if pctg else x, precision) for x in arr]


feature_and_metadata = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    xmol_clean_author_name, xmol_author_pinyin_name, pid_ao1, pid_ao2, xmol_aid1, xmol_raw_citation1, \
    coauthors1, author_affiliations1, author_countries1, author_provinces1, author_cities1, author_postcodes1, \
    en_infersent_embedding1, en_sent2vec_embedding1, en_specter_embedding1, jd1, st1, \
    title1, journal_title1, pub_year1, xmol_aid2, xmol_raw_citation2, \
    coauthors2, author_affiliations2, author_countries2, author_provinces2, author_cities2, author_postcodes2, \
    en_infersent_embedding2, en_sent2vec_embedding2, en_specter_embedding2, jd2, st2, \
    title2, journal_title2, pub_year2, commonness, train1_test0_val2 = row

    # metadata: xmol_clean_author_name, pid1, pid2, xmol_aid1, xmol_aid2, train1_test0_val2
    same_author = 1 if xmol_aid1 == xmol_aid2 else 0

    sim_coauthors_intersection = intersection(coauthors1, coauthors2)
    sim_affiliations_intersection = intersection(author_affiliations1, author_affiliations2)
    sim_country_intersection = intersection(author_countries1, author_countries2)
    sim_provinces_intersection = intersection(author_provinces1, author_provinces2)
    sim_cities_intersection = intersection(author_cities1, author_cities2)
    sim_postcode_intersection = intersection(author_postcodes1, author_postcodes2)
    jd_intersection = intersection(jd1, jd2)
    st_intersection = intersection(st1, st2)
    sim_embd_infsent = cosine_similarity(en_infersent_embedding1, en_infersent_embedding2)
    sim_embd_sent2vec = cosine_similarity(en_sent2vec_embedding1, en_sent2vec_embedding2)
    sim_embd_specter = cosine_similarity(en_specter_embedding1, en_specter_embedding2)
    # sim_embd_medbert = cosine_similarity(zh_medbert_embedding1, zh_medbert_embedding2)
    sim_embd_medbert = -1
    sim_title_jaccard = jaccard_similarity(title1.split(' '), title2.split(' '))
    sim_journal_jaccard = jaccard_similarity(journal_title1.split(' '), journal_title2.split(' '))
    sim_pubyear_diff = abs(int(pub_year1) - int(pub_year2)) if abs(int(pub_year1) - int(pub_year2)) < 40 else 0
    lastname_len = len(xmol_author_pinyin_name.split(' ')[0])

    feature_and_metadata.append(
        [xmol_clean_author_name, pid_ao1, pid_ao2, same_author, train1_test0_val2,  # [0-4]
         commonness,  # 0
         sim_coauthors_intersection,  # 1
         sim_affiliations_intersection,  # 2
         sim_country_intersection,  # 3
         sim_provinces_intersection,  # 4
         sim_cities_intersection,  # 5
         sim_postcode_intersection,  # 6
         jd_intersection,  # 7
         st_intersection,  # 8
         sim_embd_infsent,  # 9
         sim_embd_sent2vec,  # 10
         sim_embd_specter,  # 11
         sim_embd_medbert,  # 12
         sim_title_jaccard,  # 13
         sim_journal_jaccard,  # 14
         sim_pubyear_diff,  # 15
         lastname_len  # 16
         ]
    )

feature_group_dict = {
    # 'CA': ['f1'],  # coauthors
    # 'AF': ['f2'],  # original_affiliations 这个特征没有加入 因为作者机构字符串在两个数据库中的表示差异非常大，不能根据字符串是否一致判断是否同一个作者
    # 'NT-PV-CT-PC': ['f3', 'f4', 'f5', 'f6'],  # nation + province + city + postcode
    'CM-PY-JD-ST': ['f0', 'f16', 'f15', 'f7', 'f8'],  # commonness + lastname_len + pub_year + jd + st

    'InferSent': ['f9'],
    'Sent2Vec': ['f10'],
    'SPECTER': ['f11'],
    # 'MedBERT': ['f12'],
    # 'TitleSim': ['f13'],

    'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕SPECTER': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f11'],
    'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕InferSent': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f9'],
    'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕Sent2Vec': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f10'],
    # 'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕MedBERT': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f12'],
    'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕TitleSim': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f13'],
    'CA⊕NT-PV-CT-PC⊕CM-PY-JD-ST⊕SPECTER⊕JT': ['f1'] + ['f3', 'f4', 'f5', 'f6'] + ['f0', 'f16', 'f15', 'f7', 'f8'] + ['f13']
                                             + ['f14']
    # Note ##########################################################
    # Note GPU对SinoPubMed消歧使用的特征如下：
    # all_features = torch.cat((commonness_lastname_len_feature, pub_year_feature, embedding_similarity_feature.unsqueeze(1), entity_features), dim=1)
    # other_entity_fields = ['jd', 'st', 'ca', 'on', 'ov', 'oc', 'op']  # [30, 30+30)
    # arrayMap(x->concat('jd', '_', x), jd),
    # arrayMap(x->concat('st', '_', x), st),
    # arrayMap(x->concat('ca', '_', x), coauthor_list),
    # arrayMap(x->concat('on', '_', x), original_aff_countries),
    # arrayMap(x->concat('ov', '_', x), original_aff_provinces),
    # arrayMap(x->concat('oc', '_', x), original_aff_cities),
    # arrayMap(x->concat('op', '_', x), original_aff_postcode)
    # Note ##########################################################
}

meta_columns = ['xmol_clean_author_name', 'pid1', 'pid2', 'same_author', 'train1_test0_val2']
features_columns = ['f' + str(i) for i in range(len(feature_and_metadata[0]) - 5)]

df = pd.DataFrame(feature_and_metadata, columns=meta_columns + features_columns)
print(df.shape)
print('original shape: ', df.shape)
df = shuffle(df)

print('#training samples: %d; #evaluation samples: %d; ' % (
    len(df[df['train1_test0_val2'] == 1]), len(df[df['train1_test0_val2'] == 2])))

# mode_names = ModelName.available_modes()
model_switch = ModelName.randomforest
model_name = model_switch.name.lower()
print('used %s model ... \n' % model_name)

models = {}
for feature_group_as_a_method, feature_group in feature_group_dict.items():
    df_copy = df.copy(deep=True)
    Y = np.array(df_copy['same_author'].astype('int'))
    X = df_copy[feature_group]
    # X = scale(X)
    data_entry = np.array([pm_ao1 + '|' + pm_ao2 for pm_ao1, pm_ao2 in df_copy[['pid1', 'pid2']].values])
    X = np.array(X).astype(np.float)

    avg_metrics = []
    # kf = KFold(n_splits=10, shuffle=True)    'name_based_features_9',  # last name ambiguity score
    # indx_split = kf.split(Y)
    # kf = GroupShuffleSplit(n_splits=10)
    # indx_split = kf.split(X, groups=df['lastname_hash_partition_for_split'].values)

    # Note the test set is not used
    indx_split = [
        ([i for i, n in enumerate(df_copy['train1_test0_val2'].values) if n == 1],
         [i for i, n in enumerate(df_copy['train1_test0_val2'].values) if n == 2])
    ]
    for train_index, test_index in indx_split:
        train_X, train_y = X[train_index], Y[train_index]
        test_X, test_y = X[test_index], Y[test_index]
        data_entry_X, data_entry_y = data_entry[test_index], data_entry[test_index]

        model, pred_y, feature_importance = use_regression(train_X, train_y, test_X, model_switch=model_switch)

        # Note 这里先不保存模型，将在后面确定好聚类参数后再保存最佳的模型
        # saved_model_file_name = '%s-method-%s-model.pkl' % (feature_group_as_a_method, model_name)
        # joblib.dump(model, os.path.join(cached_dir, 'CHAND-models', saved_model_file_name))
        # models.append([feature_group_as_a_method, model_name, model])
        models[feature_group_as_a_method] = model

        # Note eval the model
        metric_dict = calc_metrics(test_y, pred_y)
        avg_metrics.append(metric_dict)

    avg_metric_vals = [np.average([item[m] for item in avg_metrics]) for m in metric_names]
    print('\t'.join([feature_group_as_a_method] + [str(round(n, 2)) for n in avg_metric_vals]))

    # print('-' * 160)

# Note #################################################################################################
# Note loading the block-based dev dataset, which is used for tuning clustering parameters

sql = r'''
select xmol_clean_author_name, xmol_author_pinyin_name, concat(toString(new_pid), '_', toString(author_position)) as pid_ao, xmol_aid, new_pid, xmol_raw_citation, commonness, coauthors_en, author_affiliations_en, author_countries_en, author_provinces_en, author_cities_en, author_postcodes_en, en_infersent_embedding, en_sent2vec_embedding, en_specter_embedding, title_en, jd, st, journal_title_en, pub_year
from and_ds_ench.CH_EN_AND_dataset ANY
         inner join (select pid_ao, toInt32(round(log2(block_size))) as commonness
                     from and_ds_ench.sinopubmed_ambiguous_author_with_nameblocking_en) using pid_ao
-- Note denoting Chinese AND dataset
where CHENAND > 0 and train1_test0_val2 in (2)
order by xmol_clean_author_name
;
'''

df_dev = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)


# Note #################################################################################################
# Note convert block to author-author feature vector and author-author 2D similarity matrix

def calc_block_author2author_features(df):
    """
    :param df: df is the input dataset, e.g., test set or dev set
    :return: groundtruths and predictions of all the blocks in the input dataset
    """
    block_author2author_features_dict = {}
    for block_name, sub_df in tqdm(df.groupby(['xmol_clean_author_name'])):
        block_authors = sub_df.values
        num_block_authors = len(block_authors)

        block_author_ground_truth = sub_df['xmol_aid'].values
        block_author_feature_and_metadata_3d = np.zeros(shape=(num_block_authors, num_block_authors, 17), dtype=np.float)

        for i in range(0, num_block_authors - 1, 1):
            for j in range(i, num_block_authors, 1):
                xmol_clean_author_name1, xmol_author_pinyin_name1, pid_ao1, xmol_aid1, pid1, xmol_raw_citation1, commonness1, \
                coauthors1, author_affiliations1, author_countries1, author_provinces1, author_cities1, author_postcodes1, \
                en_infersent_embedding1, en_sent2vec_embedding1, en_specter_embedding1, \
                title1, jd1, st1, journal_title1, pub_year1 = block_authors[i]

                xmol_clean_author_name2, xmol_author_pinyin_name2, pid_ao2, xmol_aid2, pid2, xmol_raw_citation2, commonness2, \
                coauthors2, author_affiliations2, author_countries2, author_provinces2, author_cities2, author_postcodes2, \
                en_infersent_embedding2, en_sent2vec_embedding2, en_specter_embedding2, \
                title2, jd2, st2, journal_title2, pub_year2 = block_authors[j]

                # metadata: xmol_clean_author_name, pid1, pid2, xmol_aid1, xmol_aid2, train1_test0_val2
                same_author = 1 if xmol_aid1 == xmol_aid2 else 0

                sim_coauthors_intersection = intersection(coauthors1, coauthors2)
                sim_affiliations_intersection = intersection(author_affiliations1, author_affiliations2)
                sim_country_intersection = intersection(author_countries1, author_countries2)
                sim_provinces_intersection = intersection(author_provinces1, author_provinces2)
                sim_cities_intersection = intersection(author_cities1, author_cities2)
                sim_postcode_intersection = intersection(author_postcodes1, author_postcodes2)
                jd_intersection = intersection(jd1, jd2)
                st_intersection = intersection(st1, st2)
                sim_embd_infsent = cosine_similarity(en_infersent_embedding1, en_infersent_embedding2)
                sim_embd_sent2vec = cosine_similarity(en_sent2vec_embedding1, en_sent2vec_embedding2)
                sim_embd_specter = cosine_similarity(en_specter_embedding1, en_specter_embedding2)
                sim_embd_medbert = -1
                sim_title_jaccard = jaccard_similarity(title1.split(' '), title2.split(' '))
                sim_journal_jaccard = jaccard_similarity(journal_title1.split(' '), journal_title2.split(' '))
                sim_pubyear_diff = abs(int(pub_year1) - int(pub_year2)) if abs(int(pub_year1) - int(pub_year2)) < 40 else 0
                lastname_len = len(xmol_author_pinyin_name1.split(' ')[0])

                # metadata: block_name, xmol_aid1, xmol_aid2, i, j,  # [0-4]
                a2a_author_feature_vec = np.array([
                    commonness1,  # 0
                    sim_coauthors_intersection,  # 1
                    sim_affiliations_intersection,  # 2
                    sim_country_intersection,  # 3
                    sim_provinces_intersection,  # 4
                    sim_cities_intersection,  # 5
                    sim_postcode_intersection,  # 6
                    jd_intersection,  # 7
                    st_intersection,  # 8
                    sim_embd_infsent,  # 9
                    sim_embd_sent2vec,  # 10
                    sim_embd_specter,  # 11
                    sim_embd_medbert,  # 12
                    sim_title_jaccard,  # 13
                    sim_journal_jaccard,  # 14
                    sim_pubyear_diff,  # 15
                    lastname_len  # 16
                ]).astype(np.float)
                block_author_feature_and_metadata_3d[i][j] = a2a_author_feature_vec
                block_author_feature_and_metadata_3d[j][i] = a2a_author_feature_vec

        block_author2author_features_dict[block_name] = [block_author_feature_and_metadata_3d, block_author_ground_truth]
    return block_author2author_features_dict


def calc_groundtruth_predictions_given_dataset(dev_set_block_author2author_features_dict, model, feature_selector):
    """
    :param df: df is the input dataset, e.g., test set or dev set
    :return: groundtruths and predictions of all the blocks in the input dataset
    """
    ground_truths_ds = []
    feature_and_metadata_ds = []
    for block_name, (block_author_feature_and_metadata_3d, block_author_ground_truth) \
            in dev_set_block_author2author_features_dict.items():
        num_block_authors = len(block_author_feature_and_metadata_3d)

        batch_a2a_author_feature_vecs = block_author_feature_and_metadata_3d[:, :, feature_selector]
        batch_a2a_author_feature_vecs = batch_a2a_author_feature_vecs.reshape(
            (num_block_authors * num_block_authors, len(feature_selector)))

        batch_a2a_distance = 1 - model.predict(batch_a2a_author_feature_vecs)
        # Note block_a2a_distance is already a symmetric matrix
        block_a2a_distance = batch_a2a_distance.reshape((num_block_authors, num_block_authors))

        feature_and_metadata_ds.append(block_a2a_distance)
        ground_truths_ds.append(block_author_ground_truth)
    return ground_truths_ds, feature_and_metadata_ds


dev_set_block_author2author_features_dict = None
# Note collect the best clustering parameter for each method
best_cluster_setting_dict = {}
for feature_group_as_a_method, feature_group in feature_group_dict.items():
    print('-' * 160)
    print(feature_group_as_a_method)
    # Note for each method (implemented by a feature group), we find its best clustering parameter and save the model
    model_selector = feature_group_as_a_method
    # model_selector = 'coauthor+affentity+aff+title+journal+pubyear'
    model = models[model_selector]
    feature_selector = [int(n.replace('f', '')) for n in feature_group]

    if dev_set_block_author2author_features_dict is None:
        dev_set_block_author2author_features_dict = calc_block_author2author_features(df_dev)
    ground_truths_ds, feature_and_metadata_ds = calc_groundtruth_predictions_given_dataset(
        dev_set_block_author2author_features_dict,
        model, feature_selector)

    # Note #################################################################################################
    # Note loading the block-based dev dataset, which is used for tuning clustering parameters
    num_blocks = len(ground_truths_ds)
    best_metric = -1
    best_cluster_setting = 0
    distance_threshold_grids = [n * 0.01 for n in range(0, 101, 10)]
    print('tuning clustering parameters ...')
    metric_tendencies = []
    for distance_threshold in tqdm(distance_threshold_grids):
        cluster_algo = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed',
                                               # linkage='single',
                                               linkage='average'
                                               )
        all_clustering_metrics = []
        for block_ground_truths, block_feature_and_metadata in zip(ground_truths_ds, feature_and_metadata_ds):
            cluster_labels = cluster_algo.fit_predict(X=block_feature_and_metadata)

            # print(block_name, len(ground_truths), len(set(ground_truths)), cluster_labels)

            # Note compare the cluster_labels with the ground truth and calculate the metrics
            block_metrics_pairwisef = paired_precision_recall_fscore(labels_true=block_ground_truths, labels_pred=cluster_labels)
            block_metrics_b3 = b3_precision_recall_fscore(labels_true=block_ground_truths, labels_pred=cluster_labels)
            all_clustering_metrics.append(
                data_precision_round(list(block_metrics_pairwisef[-3:] + block_metrics_b3[-3:]), pctg=False))

        # Note computer average metrics
        avg_metrics = np.array([n for n in all_clustering_metrics]).mean(axis=0)
        # print(avg_metrics)
        pp, pr, pf, bp, br, bf = avg_metrics
        metric_tendencies.append([distance_threshold, bf])

        if best_metric < bf:
            print('updated the best clustering B3-F1 metric from %f to %f, and the the corresponding clustering setting is %f' % (
                best_metric, bf, distance_threshold))
            best_metric = bf
            best_cluster_setting = distance_threshold

    plt.plot([n[0] for n in metric_tendencies], [n[1] for n in metric_tendencies])
    plt.title(model_selector)
    plt.savefig(os.path.join(cached_dir, 'cluster_parameter_tuning/CH-%s.png' % model_selector), dpi=300)
    plt.show()

    best_cluster_setting_dict[feature_group_as_a_method] = best_cluster_setting

    # Note save the model while saving the best clustering parameter
    saved_model_file_name = '%s-method-%s-model-clusteringparams-%f.pkl' % (model_selector, model_name, best_cluster_setting)
    joblib.dump(model, os.path.join(cached_dir, 'CHENAND-models', saved_model_file_name))

    print('the best_cluster_setting for current_model: %s is %f' % (model_selector, best_cluster_setting))

# Note ##########################################################################################
# Note evaluating on the test set using best clustering parameter

print('evaluating on the test set using best clustering parameter ... ')

sql = r'''
select xmol_clean_author_name, xmol_author_pinyin_name, concat(toString(new_pid), '_', toString(author_position)) as pid_ao, xmol_aid, new_pid, xmol_raw_citation, commonness, coauthors_en, author_affiliations_en, author_countries_en, author_provinces_en, author_cities_en, author_postcodes_en, en_infersent_embedding, en_sent2vec_embedding, en_specter_embedding, title_en, jd, st,  journal_title_en, pub_year
from and_ds_ench.CH_EN_AND_dataset ANY
         inner join (select pid_ao, toInt32(round(log2(block_size))) as commonness
                     from and_ds_ench.sinopubmed_ambiguous_author_with_nameblocking_en) using pid_ao
-- Note denoting Chinese AND dataset
where CHENAND > 0 and train1_test0_val2 in (0) -- Note eval on test set
order by xmol_clean_author_name
;
'''

df_block_test = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)
df_pairwise_test = df[df['train1_test0_val2'] == 0]

test_set_block_author2author_features_dict = None
# Note collect the best clustering parameter for each method
for feature_group_as_a_method, feature_group in feature_group_dict.items():
    # Note for each method (implemented by a feature group), we use its best clustering parameter to evaluate on the TEST SET
    print('-' * 160)
    best_cluster_setting = best_cluster_setting_dict[feature_group_as_a_method]
    model_selector = feature_group_as_a_method

    # model_selector = 'coauthor+affentity+aff+title+journal+pubyear'
    model = models[model_selector]
    feature_selector = [int(n.replace('f', '')) for n in feature_group]

    if test_set_block_author2author_features_dict is None:
        test_set_block_author2author_features_dict = calc_block_author2author_features(df_block_test)
    # Note evaluated on test set df_block_test
    ground_truths_ds, feature_and_metadata_ds = calc_groundtruth_predictions_given_dataset(
        test_set_block_author2author_features_dict,
        model, feature_selector)

    cluster_algo = AgglomerativeClustering(n_clusters=None, distance_threshold=best_cluster_setting, affinity='precomputed',
                                           # linkage='single',
                                           linkage='average'
                                           )

    all_clustering_metrics = []
    for block_ground_truths, block_feature_and_metadata in zip(ground_truths_ds, feature_and_metadata_ds):
        cluster_labels = cluster_algo.fit_predict(X=block_feature_and_metadata)

        # print(block_name, len(ground_truths), len(set(ground_truths)), cluster_labels)

        # Note compare the cluster_labels with the ground truth and calculate the metrics
        block_metrics_pairwisef = paired_precision_recall_fscore(labels_true=block_ground_truths, labels_pred=cluster_labels)
        block_metrics_b3 = b3_precision_recall_fscore(labels_true=block_ground_truths, labels_pred=cluster_labels)
        all_clustering_metrics.append(['', len(block_ground_truths)] + list(block_metrics_pairwisef[-3:] + block_metrics_b3[-3:]))

    # note calculate the P R F1 metrics on PAIRWISE metrics
    ground_truths = df_pairwise_test['same_author'].values.astype('int')
    selected_columns = feature_group
    predictions = model.predict(df_pairwise_test[selected_columns].values.astype(np.float))
    pairwise_metrics = calc_metrics(ground_truths, predictions)
    pairwise_metrics = [pairwise_metrics[n] for n in metric_names]
    num_pairwise_instances = len(df_pairwise_test)

    # note calculate the paired-F1 and the B3-F1 score on BLOCK dataset
    columns = ['Block', 'BlockSize', 'pP', 'pR', 'pF', 'bP', 'bR', 'bF']
    df_metric = pd.DataFrame(all_clustering_metrics, columns=columns)
    num_block_instances = sum(df_metric['BlockSize'].values)
    del df_metric['Block'], df_metric['BlockSize']
    # note calculate the paired-F1 and the B3-F1 score on BLOCK dataset
    block_metrics = df_metric._get_numeric_data().mean().values.tolist()

    # print(mean_metrics)
    print('\t'.join(metric_names + columns))
    metric_str = '\t'.join(['ML-' + feature_group_as_a_method, str(num_pairwise_instances) + '_' + str(num_block_instances)] + [
        str(round(n * 100, 2)) for n in pairwise_metrics + block_metrics])
    print(metric_str)
    save_result(spec='CHENAND-Offline.txt', metrics=metric_str)
