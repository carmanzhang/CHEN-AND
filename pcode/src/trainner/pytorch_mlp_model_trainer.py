import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mytookit.data_reader import DBReader
from sklearn.metrics import confusion_matrix, classification_report

from model.and_model import MLP
from myconfig import cached_dir

choices = ['model_for_disambiguating_SinoMed', 'model_for_disambiguating_SinoPubMed']

choice = choices[1]

# # Note save the AND model
model_path = os.path.join(cached_dir,
                          'GPU-CHENAND-models/gpu-and-model-mlp-for-sinomed.pkl'
                          if choice == choices[0]
                          else
                          'GPU-CHENAND-models/gpu-and-model-mlp-for-sinopubmed.pkl')

sql = """
    select new_pid1, new_pid2, features, ground_truth, train1_test0_val2
    from and_ds_ench.CHENAND_dataset_sampled_author_pair_for_training_GPU_CHAND_model
    """ if choice == choices[0] else """
    select new_pid1, new_pid2, features, ground_truth, train1_test0_val2
    from and_ds_ench.CHENAND_dataset_sampled_author_pair_for_training_GPU_CHENAND_model
    """


def customized_BCELoss(output, target, pos_weight, neg_weight):
    output = torch.clamp(output, min=1e-8, max=1 - 1e-8)
    loss = pos_weight * (target * torch.log(output)) + neg_weight * ((1 - target) * torch.log(1 - output))
    return loss


def train_validate(X, Y, X_test, Y_test, model, args):
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    N = len(Y)
    losses = []
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()

    model.train()
    for epoch in range(args.epoch):
        sum_loss = 0

        perm = torch.randperm(N)
        for i in range(0, N, args.batchsize):
            x = X[perm[i: i + args.batchsize]].to(args.device)
            y = Y[perm[i: i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x)

            # loss = criterion(output, y)
            loss = criterion(output, y)
            # loss = customized_BCELoss(y, output)
            # if i % 100 == 0:
            #     print(loss.item())
            loss.backward()

            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))
        losses.append(sum_loss / N)
        validate(X_test, Y_test, model, args)
    plt.plot(losses)
    plt.show()


def validate(X, Y, model, args):
    X = torch.FloatTensor(X)
    N = len(X)
    model.eval()
    outputs = model(X.to(args.device))
    # outputs = []
    # for epoch in range(args.epoch):
    #     perm = torch.randperm(N)
    #     for i in range(0, N, args.batchsize):
    #         x = X[perm[i : i + args.batchsize]].to(args.device)
    #         output = model(x).squeeze()
    #         outputs.append(output)
    # outputs = torch.cat(outputs, dim=0).detach().cpu().numpy()

    Y = [1 if n > 0.5 else 0 for n in Y]
    outputs = [1 if n > 0.5 else 0 for n in outputs]
    print(confusion_matrix(Y, outputs))
    # tn, fp, fn, tp = confusion_matrix(Y, outputs).ravel()
    # print(tn, fp, fn, tp)
    print(classification_report(Y, outputs))

    return outputs


def visualize(X, Y, model):
    W = model.weight.squeeze().detach().cpu().numpy()
    b = model.bias.squeeze().detach().cpu().numpy()

    delta = 0.001
    x = np.arange(X[:, 0].min(), X[:, 0].max(), delta)
    y = np.arange(X[:, 1].min(), X[:, 1].max(), delta)
    x, y = np.meshgrid(x, y)
    xy = list(map(np.ravel, [x, y]))

    z = (W.dot(xy) + b).reshape(x.shape)
    z[np.where(z > 1.0)] = 4
    z[np.where((z > 0.0) & (z <= 1.0))] = 3
    z[np.where((z > -1.0) & (z <= 0.0))] = 2
    z[np.where(z <= -1.0)] = 1

    plt.figure(figsize=(10, 10))
    plt.xlim([X[:, 0].min() + delta, X[:, 0].max() - delta])
    plt.ylim([X[:, 1].min() + delta, X[:, 1].max() - delta])
    plt.contourf(x, y, z, alpha=0.8, cmap="Greys")
    plt.scatter(x=X[:, 0], y=X[:, 1], c="black", s=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)

    df = DBReader.tcp_model_cached_read("xxx", sql=sql, cached=False)
    df_train = df[df['train1_test0_val2'] == 1]
    df_val = df[df['train1_test0_val2'] == 2]
    df_test = df[df['train1_test0_val2'] == 0]
    print('df_train, df_val, df_test', df_train.shape, df_val.shape, df_test.shape)
    X_train = np.array(df_train['features'].values.tolist(), dtype=np.float32)
    X_val = np.array(df_val['features'].values.tolist(), dtype=np.float32)
    X_test = np.array(df_test['features'].values.tolist(), dtype=np.float32)

    # Y_train = np.array(df_train['ground_truth'].values.tolist(), dtype=np.float32)
    # Y_test = np.array(df_test['ground_truth'].values.tolist(), dtype=np.float32)

    Y_train = df_train['ground_truth'].values.tolist()
    Y_val = df_val['ground_truth'].values.tolist()
    Y_test = df_test['ground_truth'].values.tolist()

    print(X_train.shape, X_val.shape, X_test.shape)

    # X, Y = make_blobs(n_samples=500, centers=2, random_state=0, cluster_std=0.4)
    # X = (X - X.mean()) / X.std()
    # Y[np.where(Y == 0)] = -1

    model = MLP(input_dim=X_train.shape[1])
    model.to(args.device)

    train_validate(X_train, Y_train, X_val, Y_val, model, args)
    print('eval on val set: ... ')
    validate(X_test, Y_test, model, args)

    # pickle.dump(model, open(model_path, 'wb'))
    # model = pickle.load(open(model_path, 'rb'))
    # print(model)

    torch.save(model, model_path)
    model = torch.load(model_path)
    print(model)

    # Y_pred = validate(X_test, Y_test, model, args)

    # print(confusion_matrix(Y_test, [1 if n > 0.5 else 0 for n in Y_pred]))
    # tn, fp, fn, tp = confusion_matrix(Y_test, [1 if n > 0.5 else 0 for n in Y_pred]).ravel()
    # print(tn, fp, fn, tp)

    # visualize(X_train, Y_train, model)
