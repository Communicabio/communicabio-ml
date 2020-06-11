import os
from flask import Flask
import transformers
from transformers.modeling_utils import BeamHypotheses
import gc
from transformers import GPT2LMHeadModel
import shutil
from flask import request
import shutil
import logging
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

app = Flask(__name__)

logging.info('Loading')
model_path = './gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelWithLMHead.from_pretrained(model_path)

gc.collect()
shutil.rmtree(model_path)

MAX_ADD_LENGTH = 50
MAX_HIST_LENGTH = 500 - MAX_ADD_LENGTH
'''
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
'''

def make_response(hist, max_hist_length=MAX_HIST_LENGTH, max_add_length=MAX_ADD_LENGTH):
    input_ids = []
    for el in hist[::-1]:
        input_ids += tokenizer.encode(tokenizer.eos_token) + tokenizer.encode(el)[::-1]
    input_ids = input_ids[:max_hist_length]
    input_ids = input_ids[::-1]
    start_len = len(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    beam_output = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + 50,
        min_length=input_ids.shape[-1] + 20,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = list(beam_output[0].cpu())[start_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)

@app.route('/', methods=['POST'])
def generate():
    params = request.json
    hist = params['history']
    return {'text': make_response(hist,
                        params.get('max_hist_length', MAX_HIST_LENGTH),
                        params.get('max_add_length', MAX_ADD_LENGTH))}

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
