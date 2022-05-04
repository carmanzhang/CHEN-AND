import logging

# Note set the log level, otherwise the log.info will not print
from myconfig import device

lg = logging.getLogger()
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

import numpy as np
import torch
import tqdm
from sentence_transformers import InputExample, losses
from sentence_transformers import SentenceTransformer
from sentence_transformers import evaluation
from sentence_transformers import models
from sentence_transformers.evaluation import SimilarityFunction
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


class ModelConfig:
    loss = 'COSIN'
    epoch = 3
    batch_size = 16
    optimizer_params = {'lr': 1e-5}
    max_seq_length = 250
    concise_vector_len = 10
    warmup_steps = 1000
    evaluation_steps = 3000

    @staticmethod
    def from_dict(d: dict):
        ModelConfig.loss = d['loss'] if 'loss' in d else ModelConfig.loss
        ModelConfig.concise_vector_len = d[
            'concise_vector_len'] if 'concise_vector_len' in d else ModelConfig.concise_vector_len
        ModelConfig.epoch = d['epoch'] if 'epoch' in d else ModelConfig.epoch
        ModelConfig.batch_size = d['batch_size'] if 'batch_size' in d else ModelConfig.batch_size
        ModelConfig.max_seq_length = d['max_seq_length'] if 'max_seq_length' in d else ModelConfig.max_seq_length
        ModelConfig.warmup_steps = d['warmup_steps'] if 'warmup_steps' in d else ModelConfig.warmup_steps
        ModelConfig.evaluation_steps = d[
            'evaluation_steps'] if 'evaluation_steps' in d else ModelConfig.evaluation_steps
        ModelConfig.optimizer_params = {'lr': d['lr']} if 'lr' in d else ModelConfig.optimizer_params
        return ModelConfig


class PreTrainedModel:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path

    def load_transformer(self):
        # max_seq_length specifies the maximum number of tokens of the input.
        # The number of token is superior or equal to the number of words of an input.
        word_embedding_model = models.Transformer(self.name_or_path)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
        return model

    def load_sentence_transformer(self):
        model = SentenceTransformer(self.name_or_path, device=device)
        return model

    def load(self):
        if not '/' in self.name_or_path:
            model = self.load_sentence_transformer()
            print('load_sentence_transformer: %s' % self.name_or_path)
        else:
            try:
                model = self.load_transformer()
                print('load_transformer: %s' % self.name_or_path)
            except Exception as e:
                model = self.load_sentence_transformer()
                print('load_sentence_transformer: %s from local file' % self.name_or_path)
        return model


class ActionProcessor:
    def __init__(self, model_name_or_path, data):
        self.model = PreTrainedModel(name_or_path=model_name_or_path).load()
        if data:
            self.df_train = data[0]
            self.df_val = data[1]
            self.df_test = data[2]

    def reload_model(self, save_model_path):
        """:param
        path the path of the fine tuned model
        """
        self.model = PreTrainedModel(name_or_path=save_model_path).load()
        print('successfully reloaded the model')
        return self

    def rebuild_model(self, concise_vector_len=None):
        """:param
        path the path of the fine tuned model
                """
        if concise_vector_len is None:
            return self

        model = self.model
        df_train = self.df_train

        #####################################
        # To determine the PCA matrix, we need some example sentence embeddings.
        # Here, we compute the embeddings for 20k random sentences from the AllNLI dataset
        print('rebuilding the model, set the output vector length to %d ...' % concise_vector_len)
        pca_train_sentences = df_train['content1'].values[0:10000]
        train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

        # Compute PCA on the train embeddings matrix
        pca = PCA(n_components=concise_vector_len)
        pca.fit(train_embeddings)
        pca_comp = np.asarray(pca.components_)

        # We add a dense layer to the model, so that it will produce directly embeddings with the new size
        dense = models.Dense(in_features=model.get_sentence_embedding_dimension(),
                             out_features=concise_vector_len, bias=False,
                             activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        model.add_module('dense', dense)
        print('rebuild the model')
        # reset the fine-tuned model to attribute
        self.model = model
        return self

    def fine_tune(self, save_model_path, model_config, df_data=None, show_progress_bar=False):
        optimizer_params, epoch, batch_size, warmup_steps, evaluation_steps \
            = model_config.optimizer_params, model_config.epoch, model_config.batch_size, model_config.warmup_steps, model_config.evaluation_steps

        model = self.model
        if df_data is None:
            df_train, df_val = self.df_train, self.df_val
        else:
            df_train, df_val, df_test = df_data
        print(df_train.columns.values)

        train_examples = [InputExample(texts=[content1, content2], label=float(label_or_score)) for
                          i, (content1, content2, label_or_score) in df_train.iterrows()]
        # Define your train dataset, the dataloader and the train loss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        loss = losses.CosineSimilarityLoss(model)
        evaluator = evaluation.EmbeddingSimilarityEvaluator(df_val['content1'].values, df_val['content2'].values,
                                                            df_val['score'].values.astype('float'),
                                                            main_similarity=SimilarityFunction.COSINE)
        # Note eval using the raw model
        print('evaluation on val set using the raw model...')
        evaluator(model)
        # Tune the model
        model.fit(train_objectives=[(train_dataloader, loss)],
                  epochs=epoch,
                  optimizer_params=optimizer_params,
                  warmup_steps=warmup_steps,
                  evaluator=evaluator,
                  evaluation_steps=evaluation_steps,
                  save_best_model=True,
                  output_path=save_model_path,
                  show_progress_bar=show_progress_bar)

        # reset the fine-tuned model to attribute
        self.model = model
        return self

    def infer(self, content_list, infer_batch_size=3200, batch_size=64):
        sent_batch = []
        sent_eb = []
        for i, v in enumerate(tqdm.tqdm(content_list)):
            if type(v) == list and len(v) == 2:
                row_id, content = v
            else:
                row_id, content = i, v
            sent_batch.append([row_id, content])
            if i % infer_batch_size == 0:
                tmp_eb = self.model.encode([n[1] for n in sent_batch], batch_size=batch_size, show_progress_bar=False)
                assert len(tmp_eb) == len(sent_batch)
                sent_eb.extend([[n[0], tmp_eb[i]] for i, n in enumerate(sent_batch)])
                sent_batch = []
        if len(sent_batch) > 0:
            tmp_eb = self.model.encode([n[1] for n in sent_batch], batch_size=batch_size, show_progress_bar=False)
            assert len(tmp_eb) == len(sent_batch)
            sent_eb.extend([[n[0], tmp_eb[i]] for i, n in enumerate(sent_batch)])
        return sent_eb

    def evaluate(self):
        content_list1 = self.df_test['content1'].values.tolist()
        content_list2 = self.df_test['content2'].values.tolist()
        sent_eb1 = [n[1] for n in self.infer(content_list1)]
        sent_eb2 = [n[1] for n in self.infer(content_list2)]

        label_or_score = self.df_test['score'].values.tolist()
        cosin_sim_scores = batch_cosin_sim_score(sent_eb1, sent_eb2)
        # label
        res = report_correlation_metrics(cosin_sim_scores, label_or_score)
        return res
