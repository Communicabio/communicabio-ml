import torch
import transformers
import os

def model_saver(model, path='model'):
    os.mkdir(path)
    dct = model.state_dict()
    for el in dct:
        torch.save(dct[el], os.path.join(path, el))

def model_loader(model_class, config_class, path):
    pass

if __name__ == '__main__':
    dir = 'model'
    model = transformers.GPT2LMHeadModel.from_pretrained(f'./ru-GPT2Like')
    model_saver(model)
