'''
Note this script will take 4.5 hours
'''
import os

import networkx as nx
import numpy as np
from mytookit.data_reader import DBReader
from tqdm import tqdm

from myconfig import cached_dir

sql = r'''
select author_name      as block_name,
       groupArray(arrayPushFront(
               arrayMap(x->x, arrayFilter(x->lengthUTF8(x) > 0, arrayConcat(co_authors,
                                                                                            author_affiliations,
                                                                                            author_provinces,
                                                                                            author_cities,
                                                                                            author_postcodes, [journal_title]))),
               pid_ao) -- Note pid_ao is new_pid + ao
           as entities) as block_author_entities
from and_ds_ench.sinomed_all_authors_addvalue
-- Note ambiguous authors is cnt > 1
where author_name in (
    select author_name
    from and_ds_ench.sinomed_all_authors_addvalue
    where lengthUTF8(author_name) > 0
    --and author_name='王伟'
    group by author_name
    having count() > 1)
group by author_name
order by count() asc
;
'''

df = DBReader.tcp_model_cached_read(cached_file_path='xxx', sql=sql, cached=False)


def make_heuristic_based_pairwise_author_judgement(author_profile1, author_profile2):
    common_ele = set(author_profile1).intersection(set(author_profile2))
    return len(common_ele)


def make_heuristic_based_author_group_judgement(author_instance_profile_2d):
    num_author_names = len(author_instance_profile_2d)
    connected_edges = []
    # connected_edges_2D = np.zeros(shape=(num_author_names, num_author_names))
    # Note make author-author similarity 2D matrix
    for i in range(num_author_names):
        for j in range(num_author_names):
            if i != j:
                num_common_ele = make_heuristic_based_pairwise_author_judgement(author_instance_profile_2d[i],
                                                                                author_instance_profile_2d[j])
                if num_common_ele > 0:
                    connected_edges.append([i, j])
                    # connected_edges_2D[i][j] = num_common_ele

    g = nx.Graph()
    nodes = list(range(num_author_names))
    g.add_nodes_from(nodes)
    g.add_edges_from([[s, e] for s, e in connected_edges])
    # g.add_edges_from([[e, s] for s, e in connected_edges])

    new_predictions = [0] * num_author_names
    author_group_label = 1
    for sub_g in nx.connected_components(g):
        sub_g = g.subgraph(sub_g)
        subgraph_nodes = sub_g.nodes()
        for k in subgraph_nodes:
            new_predictions[k] = author_group_label
        author_group_label += 1
    return new_predictions


# selected_pid_aos = (
# '2008011847_1', '2012672154_1', '2013145235_1', '2012563945_4', '2013325231_2', '2013337890_2', '2013371866_2', '2013390870_2',
# '2013390872_3', '2013390873_3', '2014119206_4', '2014177406_3', '2014189167_3', '2014270420_2', '2014285260_1', '2014564326_3',
# '2015123158_1', '2015374165_2', '2015394464_2', '2015395294_4', '2015458353_4', '2015458465_1', '2015487072_2', '2015531319_2',
# '2016108721_2', '2016243711_2', '2016279102_2', '2016347235_2', '2016546654_6', '2016546657_2', '2017136967_2', '2017226309_2',
# '2017251008_4', '2017387197_6', '2017707240_2', '2018220340_2', '2018284749_2', '2018366565_1', '2018372155_2', '2018468003_5',
# '2018529744_6', '2005732693_1', '2011292849_9', '2013279949_1', '2008128148_1', '2010499104_1', '2017379686_5', '2015852020_2')

fw = open(os.path.join(cached_dir, 'sinomed-disambiguated-authors-using-heuristics.tsv'), 'w')
all_metrics = []
# Note evaluation on the test set: df_test
for i, (block_name, block_author_entities) in tqdm(df.iterrows(), total=len(df)):
    # block_author_entities = [n for n in block_author_entities if n[0] in selected_pid_aos]
    # block_author_entities = [n[0] for n in sorted([[n, dict(zip(list(selected_pid_aos), list(range(len(selected_pid_aos)))))[n[0]]] for n in block_author_entities],
    #                                key=lambda x:x[1], reverse=False)]

    pid_aos = [n[0] for n in block_author_entities]
    block_author_entities = [n[1:] for n in block_author_entities]

    predictions = make_heuristic_based_author_group_judgement(block_author_entities)
    for pid_ao, label in zip(pid_aos, predictions):
        fw.write('\t'.join([pid_ao, str(label)]))
        fw.write('\n')
