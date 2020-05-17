import os
from flask import Flask
import transformers
import gc
from tokenizer import RubertaTokenizer
import shutil
from flask import request
import torch
import shutil
import logging
#from google.cloud import error_reporting

#client = error_reporting.Client()
#os.system('nvidia-smi >file.txt')
#client.report(open('file.txt').read())
#assert(torch.cuda.device_count() >= 1)

app = Flask(__name__)

'''logging.info('Loading')
tokenizer = RubertaTokenizer('vocab.bpe')
device = torch.device('cuda:0')
model = transformers.GPT2LMHeadModel.from_pretrained('./ru-GPT2Like').to(device) #utils.load_model(f'./ru-GPT2Like')
logging.info('Model done')
gc.collect()

MAX_ADD_LENGTH = 70
MAX_HIST_LENGTH = 500 - MAX_ADD_LENGTH
EOS_ID = 50000

def make_response(hist, max_hist_length=MAX_HIST_LENGTH, max_add_length=MAX_ADD_LENGTH):
    with torch.no_grad():
        input_ids = []
        for el in hist[::-1]:
            input_ids += [EOS_ID] + tokenizer.encode(el)[::-1]
        input_ids = input_ids[:max_hist_length]
        input_ids = input_ids[::-1]
        start_len = len(input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        beam_output = model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + max_add_length,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        generated = list(beam_output[0].cpu())[start_len:]
        for i in range(len(generated)):
            if generated[i] == EOS_ID:
                return tokenizer.decode(generated[:i])
        return tokenizer.decode(generated)

@app.route('/', methods=['POST'])
def generate():
    params = request.json
    hist = params['history']
    return {'text': make_response(hist,
                        params.get('max_hist_length', MAX_HIST_LENGTH),
                        params.get('max_add_length', MAX_ADD_LENGTH))}'''

@app.route('/hello')
def hello_world():
    return 'hello, world!'

@app.route('/')
def check_gpu():
    code = os.system('nvidia-smi >file.txt')
    assert(code != 0)
    return open('file.txt').read()

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
