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

class Bert4Classification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = embedlib.models.RoBERTaLike(lang='ru',
                                                head_lays=0,
                                                models=['qembedder'])
        self.classifier = torch.nn.Sequential(torch.nn.Linear(768, 1),
                                            torch.nn.Sigmoid())

    def forward(self, X):
        embeddings = self.model.qembedd(X)
        return self.classifier(embeddings)

    def save_to(self, dir):
        os.mkdir(dir)
        torch.save(self.classifier, os.path.join(dir, 'classifier.bin'))
        path = os.path.join(dir, 'model')
        os.mkdir(path)
        self.model.save_to(path)

    def load_from(self, dir):
        self.model = embedlib.utils.load_model(os.path.join(dir, 'model'))
        self.classifier = torch.load(os.path.join(dir, 'classifier.bin'))


class LiveJournalInsults(torch.utils.data.Dataset):
    """http://tpc.at.ispras.ru/prakticheskoe-zadanie-2015/"""
    path = 'discussions_tpc_2015'

    def recursive_search(self, item):
        is_insult = item.get('insult', False)
        if is_insult:
            self.insults.append(item['text'])
        else:
            self.normal.append(item['text'])

        for child in item.get('children', []):
            self.recursive_search(child)

    def __init__(self, path=path):
        self.insults = []
        self.normal = []
        for el in os.listdir(path):
            for file in os.listdir(os.path.join(path, el)):
                data = json.load(open(os.path.join(path, el, file)))
                for entrie in data:
                    self.recursive_search(entrie['root'])
        mn = min(len(self.insults), len(self.normal))
        insults = self.insults[:mn]
        normal = self.normal[:mn]
        self.data = [(txt, 0.0) for txt in insults] + [(txt, 1.0) for txt in normal]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

device = torch.device('cpu') if torch.cuda.device_count() == 0 else torch.device('cuda:0')
model = Bert4Classification()
model.to(device)

dataset = LiveJournalInsults()
print(f"Dataset size: {len(dataset)}")
test_size = int(len(dataset) * 0.3)
train_size = len(dataset) - test_size
train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size,
                                      shuffle=True) #   collate_fn=collate_wrapper
test_loader = DataLoader(test_data, batch_size=batch_size,
                                    shuffle=True)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
criterion = torch.nn.BCELoss()
checkpoint_dir = 'insult-checkpoints/'
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
        # print(texts)
        # print(labels)
        labels = labels.to(device, dtype=torch.float)
        texts = list(texts)
        predicted = model(texts).view(-1)
        loss = criterion(predicted, labels)
        if dtype == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        total += labels.shape[-1]
        curr_right = 0
        for i in range(labels.shape[-1]):
            if labels[i] == 0 and predicted[i] < 0.5:
                curr_right += 1
            elif labels[i] == 1 and predicted[i] >= 0.5:
                curr_right += 1
        right += curr_right
        writer.add_scalar(f"{dtype}/score", curr_right/labels.shape[-1], step_num)
        writer.add_scalar(f"{dtype}/loss", loss.item(), step_num)
        step_num += 1

    return total_loss/total, right/total

writer = SummaryWriter()

model.train()
for epoch in range(50):
    train_mean_loss, train_acc = run(train_loader, dtype='train', epoch=epoch)
    print(f"train_mean_loss: {train_mean_loss:9.4f} train_acc: {train_acc:9.4f}")
    with torch.no_grad():
        test_mean_loss, test_acc = run(test_loader, dtype='test', epoch=epoch)
        print(f"test_mean_loss: {test_mean_loss:9.4f} test_acc: {test_acc:9.4f}")
    model_name = f"{test_acc:9.4f}|{test_mean_loss:9.4f}".replace(' ', '_')
    model.save_to(os.path.join(checkpoint_dir, model_name))
Bert4Classification.load_from(os.path.join(checkpoint_dir, model_name))
