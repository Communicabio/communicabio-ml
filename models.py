import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import os
import gc
import logging

MODEL_PATH = 'pretrained_models'

class BERTLike(torch.nn.Module):
    """Based on DeepPavlov`s BERT"""
    add_net = None
    max_seq = 512

    def __init__(self, path=None, model=None, net=None, classifier=None):
        super().__init__()

        if model is None:
            if path is None:
                path = os.path.join(MODEL_PATH, 'rubert-base-uncased')
            self.model = BertModel.from_pretrained(path)
            self.tokenizer = BertTokenizer.from_pretrained(path)
        else:
            self.model = model
            self.tokenizer = tokenizer
        gc.collect()

        if classifier is None:
            self.net = self.add_net()
        else:
            self.net = net

    def get_device(self):
        return next(self.parameters()).device

    def embed_text(self, encoded_text):
        with torch.no_grad():
            vector = self.model(torch.tensor([encoded_text],
                                             device=self.get_device()))[0]
            vector = torch.sum(vector, dim=1)
            vector = F.normalize(vector, dim=1)[0]
            return vector

    def embed(self, X):
        embeddings = [self.embed_text(self.tokenizer.encode(text,
                                                            add_special_tokens=True,
                                                            max_length=self.max_seq)) \
                      .view(1, -1) for text in X]
        return torch.cat(embeddings)

    def forward(self, X):
        with torch.no_grad():
            embeddings = self.embed(X)
            assert(embeddings.shape[0] == len(X))
        return self.net(embeddings)

    def save_to(self, dir):
        os.mkdir(dir)
        torch.save(self.net, os.path.join(dir, 'net.pt'))
        path = os.path.join(dir, 'model')
        os.mkdir(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load_from(cls, dir):
        path = os.path.join(dir, 'model')
        model = BertModel.from_pretrained(path)
        tokenizer = BertModel.from_pretrained(path)
        net = torch.load(os.path.join(dir, 'net.pt'))
        return cls(model=model, net=net, tokenizer=tokenizer)

    def train(self, flag=True):
        if flag:
            self.model.eval()
            self.net.train()
        else:
            self.model.eval()
            self.net.eval()

class Bert3Classes(BERTLike):
    add_net = lambda self: torch.nn.Sequential(torch.nn.Dropout(0.3),
                                          torch.nn.Linear(768, 3),
                                          torch.nn.Softmax(dim=-1))


class BertBinary(BERTLike):
    add_net = lambda self: torch.nn.Sequential(torch.nn.Dropout(0.3),
                                          torch.nn.Linear(768, 1),
                                          torch.nn.Sigmoid())

class BertRegression(BERTLike):
    add_net = lambda self: torch.nn.Sequential(torch.nn.Dropout(0.3),
                                        torch.nn.Linear(768, 1),
                                        torch.nn.Sigmoid())
