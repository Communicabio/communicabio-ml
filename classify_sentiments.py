import os
import json
import random
import gc
import tqdm
import csv

import transformers
import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

import embedlib
from embedlib.datasets import collate_wrapper

class Bert4SentimentClassification(torch.nn.Module):
    def __init__(self, model=None, classifier=None):
        super().__init__()
        if model is None:
            self.model = embedlib.models.BERTLike(lang='ru',
                                                  head_lays=0,
                                                  models=['qembedder'],
                                                  bert_type='bert-base-uncased')
        else:
            self.model = model
        gc.collect()

        if classifier is None:
            self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(768, 3),
                                                    torch.nn.Softmax(dim=-1))
        else:
            self.classifier = classifier

    def forward(self, X):
        with torch.no_grad():
            embeddings = self.model.qembedd(X)
        return self.classifier(embeddings)

    def save_to(self, dir):
        os.mkdir(dir)
        torch.save(self.classifier, os.path.join(dir, 'classifier.bin'))
        path = os.path.join(dir, 'model')
        os.mkdir(path)
        self.model.save_to(path)

    @classmethod
    def load_from(cls, dir):
        model = embedlib.utils.load_model(os.path.join(dir, 'model'))
        classifier = torch.load(os.path.join(dir, 'classifier.bin'))
        return cls(model=model, classifier=classifier)

    def train(self, flag=True):
        if flag:
            self.model.eval()
            self.classifier.train()
        else:
            self.model.eval()
            self.classifier.eval()

class RuSentimentDataset(torch.utils.data.Dataset):
    """https://gitlab.com/kensand/rusentiment"""
    path = 'rusentiment'

    def __init__(self, path=path):
        self.data = []
        class2id = {
            'neutral': 0,
            'negative': 1,
            'positive': 2,
        }
        for file in os.listdir(path):
            reader = csv.reader(open(os.path.join(path, file)))
            for row in reader:
                if row[0] not in class2id:
                    continue
                self.data.append((row[1], class2id[row[0]]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

if __name__ == '__main__':
    device = torch.device('cpu') if torch.cuda.device_count() == 0 else torch.device('cuda:0')
    model = Bert4SentimentClassification()
    model.to(device)

    dataset = RuSentimentDataset()
    print(f"Dataset size: {len(dataset)}")
    test_size = int(len(dataset) * 0.3)
    train_size = len(dataset) - test_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size = 16
    train_loader = DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True) #   collate_fn=collate_wrapper
    test_loader = DataLoader(test_data, batch_size=batch_size,
                                        shuffle=True)

    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    checkpoint_dir = 'sentiment-checkpoints/'
    try:
        os.mkdir(checkpoint_dir)
    except:
        pass

    step_num = 0
    def run(loader, dtype='train', epoch=0):
        global step_num
        total_loss = 0
        right = 0
        total = 0
        for texts, labels in tqdm.tqdm(iter(loader), desc=f"{dtype} epoch {epoch}"):
            labels = labels.to(device)
            texts = list(texts)
            predicted = model(texts)
            loss = criterion(predicted, labels)
            if dtype == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total += labels.shape[-1]
            curr_right = 0
            for i in range(labels.shape[-1]):
                curr_right += (max(predicted[i][0], predicted[i][1], predicted[i][2]) == predicted[i][labels[i]]).item()
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

        with torch.no_grad():
            model.eval()
            test_mean_loss, test_acc = run(test_loader, dtype='test', epoch=epoch)
            print(f"test_mean_loss: {test_mean_loss:9.4f} test_acc: {test_acc:9.4f}")
        model_name = f"{test_acc:9.4f}|{test_mean_loss:9.4f}".replace(' ', '_')
        model.save_to(os.path.join(checkpoint_dir, model_name))
    Bert4SentimentClassification.load_from(os.path.join(checkpoint_dir, model_name))
