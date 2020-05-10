import os
from flask import Flask
import transformers
import gc
from tokenizer import RubertaTokenizer
import shutil
from flask import request
import torch

app = Flask(__name__)

model_dir = './ru-GPT2Like'
tokenizer = RubertaTokenizer(os.path.join(model_dir, 'vocab.bpe'))
model = transformers.GPT2LMHeadModel.from_pretrained(f'./ru-GPT2Like')
print('Model done')
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
                        params.get('max_add_length', MAX_ADD_LENGTH))}

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
