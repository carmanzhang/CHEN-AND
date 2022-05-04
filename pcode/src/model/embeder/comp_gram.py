import os

import sent2vec

from myconfig import pretrained_model_path

"""
Note please refer to https://github.com/epfml/sent2vec
and Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features, NAACL, 2018
"""


class Sent2VecModel():
    def __init__(self, model_name):
        # Note add a method_signature
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        model = sent2vec.Sent2vecModel()
        model_path = os.path.join(pretrained_model_path, self.model_name + '.bin')
        model.load_model(model_path)
        self.model = model

    def infer_embedding(self, content_list, num_threads=-1):
        if self.model is None:
            self._load_model()
        return self.model.embed_sentences(content_list, num_threads=num_threads)


if __name__ == '__main__':
    Sent2VecModel(model_name='WikiSentVec_wiki_unigrams').infer_embedding(['hello'])
    Sent2VecModel(model_name='BioSentVec_PubMed_MIMICIII-bigram_d700').infer_embedding(['hello'])
