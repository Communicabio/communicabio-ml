import os
from flask import Flask
from flask import request
import transformers
import gc
import torch
import pickle
import json
import torch.nn.functional as F
import shutil
import os

app = Flask(__name__)

model_dir = './rubert-base-uncased'
print(os.listdir(model_dir))
tokenizer = transformers.BertTokenizer.from_pretrained(model_dir)
model = transformers.BertModel.from_pretrained(model_dir)
gc.collect()

@app.route('/', methods=['POST'])
def generate():
    with torch.no_grad():
        params = request.json
        text = params['text']
        vector = model(torch.tensor([tokenizer.encode(text, add_special_tokens=True)]))[0]
        vector = torch.sum(vector, dim=1)
        vector = F.normalize(vector, dim=1)[0]
        vector = [el.item() for el in list(vector)]
        print(vector, type(vector))
        return json.dumps({'vector': vector})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
