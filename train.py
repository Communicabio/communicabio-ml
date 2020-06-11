import sacred
from sacred import Experiment
from sacred.observers import MongoObserver
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import logging
import json

import models
import datasets
import metrics

ex = Experiment()

if 'mongo.json' in os.listdir('.'):
    url = json.load(open('mongo.json'))['url']
    ex.observers.append(MongoObserver(url=url,
                                      db_name='experiments',
                                      port=27017))
else:
    logging.warning("No observer is selected. Add mongo.json to add MongoObserver")

@ex.config
def my_config():
    model_name = 'BertBinary'
    model_path = None
    dataset = 'BlackmoonInsults'
    dataset_path = None

    metric = 'binary_accuracy'
    criterion = 'BCELoss'

    batch_size = 16
    val_ratio = .3
    lr = 1e-4
    momentum = 0.9
    epochs = 10
    checkpoint_dir = 'checkpoints'
    accumulator = 25
    device = torch.device('cpu') if torch.cuda.device_count() == 0 else torch.device('cuda:0')

@ex.capture
def load_model(model_name, model_path, device):
    return getattr(models, model_name)(model_path).to(device)

@ex.capture
def get_data(dataset, dataset_path, val_ratio, batch_size, _log):
    if dataset_path is not None:
        dataset = getattr(datasets, dataset)(dataset_path)
    else:
        dataset = getattr(datasets, dataset)()
    test_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - test_size
    train_data, test_data = torch.utils.data.dataset.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader

@ex.capture
def get_optimizer(model, lr, momentum):
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

@ex.capture
def get_criterion(criterion):
    return getattr(torch.nn, criterion)()

@ex.capture
def get_metric(metric):
    return getattr(metrics, metric)

@ex.capture
def train_epoch(model, optimizer, loader, mode, epoch, device, accumulator, _log, _run):
    criterion = get_criterion()
    metric = get_metric()
    n = 0
    total_mean_loss = total_mean_score = 0
    acc_mean_loss = acc_mean_score = acc_n = 0

    for labels, texts in tqdm(loader, desc=f"epoch {epoch:2}|{mode:5}"):
        results = model(texts).view(-1)
        labels = labels.to(device, dtype=torch.float)

        loss = criterion(results, labels)
        score = metric(texts, results, labels)

        total_mean_loss += loss.item()
        total_mean_score += score

        acc_n += len(texts)
        acc_mean_loss += loss.item()
        assert(not torch.isnan(loss))
        acc_mean_score += score

        n += 1
        if n == accumulator:
            _run.log_scalar(f"{mode}.loss", acc_mean_loss / acc_n)
            _run.log_scalar(f"{mode}.score", acc_mean_score / accumulator)
            n = 0
            acc_n = acc_mean_loss = acc_mean_score = 0

        if mode != 'test':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    total_mean_loss /= len(loader)
    total_mean_score /= len(loader)

    _run.log_scalar(f"{mode}.epoch_loss", total_mean_loss)
    _run.log_scalar(f"{mode}.epoch_score", total_mean_score)

    return total_mean_loss, total_mean_score

@ex.automain
def train(epochs, checkpoint_dir, model_name, metric, criterion):
    train_loader, test_loader = get_data()
    model = load_model()
    optimizer = get_optimizer(model)
    try:
        os.mkdir(checkpoint_dir)
    except FileExistsError:
        ...

    for epoch in range(epochs):
        model.train()
        _ = train_epoch(model, optimizer, train_loader, 'train', epoch)
        model.eval()
        with torch.no_grad():
            loss, score = train_epoch(model, optimizer, test_loader, 'test', epoch)
        name =  f"{model_name}-{metric}-{score:6.4f}-{criterion}-{loss:6.4f}"
        model.save_to(os.path.join(checkpoint_dir, name))
