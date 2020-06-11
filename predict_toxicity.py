import os
import json
import random
import gc
import tqdm

import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

import embedlib
from embedlib.datasets import collate_wrapper
import _tokenizers
import torch.nn.functional as F

from transformers import DistilBertTokenizer, DistilBertModel

import csv

device = torch.device('cuda:0')

max_seq_len = 500
class Bert4Regression(torch.nn.Module):
    def __init__(self, model=None, tokenizer=None, classifier=None):
        super().__init__()
        if model is None:
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-cased')
            # self.model.qembedder.encoder.layer = self.model.qembedder.encoder.layer[:-2]
            # self.model = transformers.BertModel.from_pretrained("ruberta") # memhacks.load_model('splitted')
            # print(self.model)
            # self.tokenizer = _tokenizers.RubertaTokenizer('vocab.bpe')
        else:
            self.model = model
            self.tokenizer = tokenizer
        gc.collect()

        if classifier is None:
            self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3),
                                                torch.nn.Linear(768, 1),
                                                torch.nn.Sigmoid())
        else:
            self.classifier = classifier

    def get_embedding(self, text):
        text = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)], device=device)
        vector = self.model(text)[0]
        vector = torch.sum(vector, dim=1)
        vector = F.normalize(vector, dim=1)[0]
        return vector.view(1, -1)

    def forward(self, X):
        embeddings = torch.cat([self.get_embedding(el) for el in X])
        return self.classifier(embeddings)

    def save_to(self, dir):
        os.mkdir(dir)
        torch.save(self.classifier, os.path.join(dir, 'classifier.bin'))

    '''@classmethod
    def load_from(cls, dir):
        model = embedlib.utils.load_model(os.path.join(dir, 'model'))
        classifier = torch.load(os.path.join(dir, 'classifier.bin'))
        return cls(model=model, classifier=classifier)'''

class JigsawDataset(torch.utils.data.Dataset):
    """https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data"""
    path = 'jigsaw.csv'

    def __init__(self, path=path):
        self.data = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            washeader = False
            for row in spamreader:
                if not washeader:
                    washeader = True
                    continue
                self.data.append((float(row[1]), row[2]))

    def __len__(self):
        return min(len(self.data), 10000)

    def __getitem__(self, ind):
        shift = 0 # 900
        return self.data[shift + ind]
T = 0

if __name__ == '__main__':

    model = Bert4Regression()
    model.to(device)

    dataset = JigsawDataset()
    print(f"Dataset size: {len(dataset)}")
    test_size = int(len(dataset) * 0.3)
    train_size = len(dataset) - test_size
    torch.manual_seed(1)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size = 16
    #train_data = train_data[:2 * batch_size]
    train_loader = DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True) #   collate_fn=collate_wrapper
    test_loader = DataLoader(test_data, batch_size=batch_size,
                                        shuffle=True)

    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    checkpoint_dir = 'insult-checkpoints/'
    try:
        os.mkdir(checkpoint_dir)
    except:
        pass

    step_num = 0
    def run(loader, dtype='train', epoch=0):
        global step_num, T
        total_loss = 0
        right = 0
        total = 0
        for labels, texts in tqdm.tqdm(iter(loader), desc=f"{dtype} epoch {epoch}"):
            texts = list(texts)
            labels = torch.tensor(labels, device=device, dtype=torch.float32)
            T += 1
            predicted = model(texts).view(-1)
            loss = criterion(predicted, labels)
            if dtype == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total += labels.shape[-1]
            curr_right = 0
            print(labels)
            print(predicted)
            print('---')
            #print(labels)
            #print(predicted)
            #print('---')
            for i in range(labels.shape[-1]):
                if abs(labels[i]) > 0.01 and abs(predicted[i] - labels[i]) <= 0.1:
                    curr_right += 1
                elif abs(labels[i]) <= .01 and predicted[i] < .01:
                    curr_right += 1

            right += curr_right
            writer.add_scalar(f"{dtype}/score", curr_right/labels.shape[-1], step_num)
            writer.add_scalar(f"{dtype}/loss", loss.item(), step_num)
            step_num += 1

        return total_loss/total, right/total

    writer = SummaryWriter()

    for epoch in range(25):
        model.train()
        train_mean_loss, train_acc = run(train_loader, dtype='train', epoch=epoch)
        print(f"train_mean_loss: {train_mean_loss:9.4f} train_acc: {train_acc:9.4f}")

        model.eval()
        with torch.no_grad():
            test_mean_loss, test_acc = run(test_loader, dtype='test', epoch=epoch)
            print(f"test_mean_loss: {test_mean_loss:9.4f} test_acc: {test_acc:9.4f}")
        model_name = f"{test_acc:9.4f}|{test_mean_loss:9.4f}".replace(' ', '_')
        model.save_to(os.path.join(checkpoint_dir, model_name))
    # Bert4Classification.load_from(os.path.join(checkpoint_dir, model_name))
