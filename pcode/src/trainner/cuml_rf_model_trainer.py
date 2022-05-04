import os
import pickle

import cudf
import joblib
import numpy as np
import pandas as pd
from mytookit.data_reader import DBReader

from eutilities.customized_print import pprint
from eutilities.metric import calc_metrics, metric_names
from model.regression import ModelName, use_regression
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from myconfig import cached_dir
from sklearn.metrics import confusion_matrix, classification_report

df = DBReader.tcp_model_cached_read("xxx",
                                    sql="""
select new_pid1, new_pid2, features, ground_truth, train1_test0_val2
from and_ds_ench.CHENAND_dataset_sampled_author_pair_for_training_GPU_CHAND_model;
""", cached=False)

df_train = df[df['train1_test0_val2'] == 1]
df_val = df[df['train1_test0_val2'] == 2]
df_test = df[df['train1_test0_val2'] == 0]

print('df_train, df_val, df_test', df_train.shape, df_val.shape, df_test.shape)

X_train = np.array(df_train['features'].values.tolist(), dtype=np.float32)
# Y_train = np.array(df_train['ground_truth'].values.tolist(), dtype=np.float32)
Y_train = df_train['ground_truth'].values.tolist()

X_test = np.array(df_test['features'].values.tolist(), dtype=np.float32)
# Y_test = np.array(df_test['ground_truth'].values.tolist(), dtype=np.float32)
Y_test = df_test['ground_truth'].values.tolist()

print(X_train.shape, X_test.shape)


# Note ########################################################################################################
# Note Training model

def cuml_RF_classifier(train_X, train_y, test_X):
    X_cudf_train = cudf.DataFrame(train_X)
    y_cudf_train = cudf.Series(train_y)

    cuml_model = curfc(n_estimators=100,
                       # max_depth=16,
                       # max_features=1.0,
                       # random_state=10
                       )

    cuml_model.fit(X_cudf_train, y_cudf_train)
    X_cudf_test = cudf.DataFrame(test_X)
    predictions = cuml_model.predict(X_cudf_test).values.get()  # using .get() converting to CPU device
    # predictions = cuml_model.predict_proba(X_cudf_test).values

    return predictions, cuml_model


def cuml_RF_regression(train_X, train_y, test_X):
    X_cudf_train = cudf.DataFrame(train_X)
    y_cudf_train = cudf.Series(train_y)

    cuml_model = curfr(n_estimators=100,
                       # max_depth=16,
                       # max_features=1.0,
                       # random_state=10
                       )

    cuml_model.fit(X_cudf_train, y_cudf_train)

    X_cudf_test = cudf.DataFrame(test_X)
    predictions = cuml_model.predict(X_cudf_test).values.get()  # using .get() converting to CPU device
    # predictions = cuml_model.predict_proba(X_cudf_test).values

    return predictions, cuml_model


# # Note cpu sklearn
# model_switch = ModelName.randomforest
# model, Y_pred, feature_importance = use_regression(X_train=X_train, Y_train=Y_train, X_test=X_test, model_switch=model_switch)
# df = pd.DataFrame([[str(round(n * 100, 4)) for n in feature_importance]], columns=feature_names).transpose()
# pickle.dump(model, open(os.path.join(cached_dir, 'cpu-and-model-%s.pkl' % model_switch.value), 'wb'))
# print(df)

# Note gpu cuml
gpu_AND_model_path = os.path.join(cached_dir, 'GPU-CHENAND-models/gpu-and-model-cumlrf-for-sinomed.pkl')
# Y_pred, model = cuml_RF_classifier(train_X=X_train, train_y=Y_train, test_X=X_test)
Y_pred, model = cuml_RF_regression(train_X=X_train, train_y=Y_train, test_X=X_test)
pickle.dump(model, open(gpu_AND_model_path, 'wb'))
# joblib.dump(model, gpu_AND_model_path)

# print(Y_pred)
Y_pred = [1 if n > 0.5 else 0 for n in Y_pred]
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print(tn, fp, fn, tp)
metrics = calc_metrics(test_y=Y_test, pred_y=Y_pred)
metric_tuple = [(m, metrics[m]) for m in metric_names]

print(metric_tuple)
print()

model = pickle.load(open(gpu_AND_model_path, 'rb'))
# print(model)
