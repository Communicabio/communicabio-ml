import torch
import csv
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

class RuSentimentDataset(Dataset):
    """https://gitlab.com/kensand/rusentiment"""
    path = 'datasets/rusentiment'

    def __init__(self, path=path):
        super().__init__()
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
                self.data.append((class2id[row[0]], row[1]))

class LiveJournalInsults(Dataset):
    """http://tpc.at.ispras.ru/prakticheskoe-zadanie-2015/"""
    path = 'datasets/discussions_tpc_2015'

    def recursive_search(self, item):
        is_insult = item.get('insult', False)
        if is_insult:
            self.insults.append(item['text'])
        else:
            self.normal.append(item['text'])

        for child in item.get('children', []):
            self.recursive_search(child)

    def __init__(self, path=path):
        super().__init__()
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
        self.data = [(.0, txt) for txt in insults] + [(1.0, txt) for txt in normal]

class JigsawDataset(Dataset):
    """https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification"""
    path = 'datasets/jigsaw.csv'

    def __init__(self, path=path):
        super().__init__()
        self.data = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            washeader = False
            for row in spamreader:
                if not washeader:
                    washeader = True
                    continue
                self.data.append((float(row[1]), row[2]))

class BlackmoonInsults(Dataset):
    """https://www.kaggle.com/blackmoon/russian-language-toxic-comments"""
    path = 'datasets/blackmoon.csv'

    def __init__(self, path=path):
        super().__init__()
        self.data = []

        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            washeader = False
            for row in spamreader:
                if not washeader:
                    washeader = True
                    continue
                self.data.append((float(row[1]), row[0].rstrip()))
