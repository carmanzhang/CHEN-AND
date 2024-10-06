import os

import joblib
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
from myconfig import cached_dir
from textwrap import wrap

title_font_size = 12

base_path = os.path.join(cached_dir, 'CHENAND-V2-models-info-collector')
model_pkl_files = [
    'NP',
    'CA',
    'AF',
    # 'NP⊕CA⊕JT',
    # 'NP⊕JT⊕PY',
    'NP⊕CA',
    'NP⊕CA⊕JT⊕PY',
    'NT-PV-CT-PC',
    'NP⊕CA⊕NT-PV-CT-PC',
    'NP⊕CA⊕NT-PV-CT-PC⊕JT',
    # 'NP⊕CA⊕NT-PV-CT-PC⊕PY',
    'NP⊕CA⊕NT-PV-CT-PC⊕JT⊕PY',
    # 'EB',
    # 'NP⊕CA⊕NT-PV-CT-PC⊕EB',
    # 'NP⊕CA⊕NT-PV-CT-PC⊕EB⊕JT⊕PY',
]

# model_pkl_files = sorted([f for f in listdir(base_path) if isfile(join(base_path, f))], key=lambda x: len(x))
# model_pkl_files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

n_rows = 3
n_cols = 3

# Note 特征重要性
fig, ax = plt.subplots(n_rows, n_cols, figsize=(9, 8))
all_features = ['NP', 'AF', 'CA', 'JD', 'ST', 'NT', 'PV', 'CT', 'PC', 'JT']
for i, model_pkl_file in enumerate(model_pkl_files):
    x_idx = i // n_cols
    y_idx = i % n_cols
    print(i, x_idx, y_idx)
    sub_ax = ax[x_idx][y_idx]
    # model_pkl_file = model_pkl_files[6]

    model_selector, model, metric_tendencies = joblib.load(os.path.join(base_path, model_pkl_file))

    # print(model.feature_importances_)
    feature_names = model_selector.replace('⊕', '-').split('-')
    print(feature_names)
    # if len(feature_names) <= 1:
    #     continue

    # feature_importances = [0] * len(all_features)
    # for j, feat in enumerate(feature_names):
    #     feature_importances[all_features.index(feat)] = model.feature_importances_[j]
    # sub_ax.bar(all_features, feature_importances, color='#6A8677', alpha=1.0, width=0.5)

    # num_features = len(feature_names)
    # num_padding_features = round((12 - num_features)/2)
    # all_features = [' '] * num_padding_features + feature_names + [' '] * num_padding_features
    # feature_importances = [0] * num_padding_features + model.feature_importances_.tolist() + [0] * num_padding_features
    # print(all_features, feature_importances, len(feature_importances))
    # sub_ax.bar(all_features, feature_importances, color='#6A8677', alpha=1.0, width=0.5)
    sub_ax.bar(feature_names, model.feature_importances_ * 100, color='#6A8677', alpha=1.0, width=0.6)
    sub_ax.set_yticklabels([str(int(x)) + '%' for x in sub_ax.get_yticks()])  # y 轴加上百分号
    feature_names = '+'.join(feature_names)
    sub_ax.set_title("\n".join(wrap(feature_names, 14)), fontsize=title_font_size)
    # sub_ax.set_ylabel('%')
    sub_ax.set_xlim(-1, 8)

plt.tight_layout()
plt.savefig(os.path.join(cached_dir, 'CHEN-AND-V2-feature-importance.png'), dpi=300)
plt.savefig(os.path.join(cached_dir, 'CHEN-AND-V2-feature-importance.pdf'), dpi=300)
plt.show()

# Note 绘制调参过程
fig, ax = plt.subplots(n_rows, n_cols, figsize=(9, 8))
for i, model_pkl_file in enumerate(model_pkl_files):
    x_idx = i // n_cols
    y_idx = i % n_cols
    print(i, x_idx, y_idx)
    sub_ax = ax[x_idx][y_idx]
    # model_pkl_file = model_pkl_files[6]
    model_selector, model, metric_tendencies = joblib.load(os.path.join(base_path, model_pkl_file))
    feature_names = model_selector.replace('⊕', '-').split('-')
    # if len(feature_names) <= 1:
    #     continue

    # metric_tendencies pp, pr, pf, bp, br, bf
    params = [n[0] for n in metric_tendencies]
    metrics = np.array([n[1] for n in metric_tendencies])

    pf = (metrics[:, 2] * 100).tolist()
    bf = (metrics[:, 5] * 100).tolist()

    max_point_index = np.argmax(bf)
    sub_ax.plot(params, bf, label='B3-F1', lw=3, marker='^')
    sub_ax.axvline(x=params[max_point_index], color='green', alpha=0.7, ls=':', lw=1)
    sub_ax.axhline(y=bf[max_point_index], color='green', alpha=0.7, ls=':', lw=1)

    max_point_index = np.argmax(pf)
    sub_ax.plot(params, pf, label='P-F1', lw=3, marker='o')
    sub_ax.axvline(x=params[max_point_index], color='green', alpha=0.7, ls=':', lw=1)
    sub_ax.axhline(y=pf[max_point_index], color='green', alpha=0.7, ls=':', lw=1)
    feature_names = '+'.join(feature_names)
    sub_ax.set_title("\n".join(wrap(feature_names, 14)), fontsize=title_font_size)
    sub_ax.set_xlim(0, 1)
    sub_ax.set_ylim(0, 100)
    sub_ax.set_yticklabels([str(int(x)) + '%' for x in sub_ax.get_yticks()])  # y 轴加上百分号

    sub_ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(cached_dir, 'CHEN-AND-V2-clustering-parameter-impact.png'), dpi=300)
plt.savefig(os.path.join(cached_dir, 'CHEN-AND-V2-clustering-parameter-impact.pdf'), dpi=300)
plt.show()
