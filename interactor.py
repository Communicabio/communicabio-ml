import pymongo
from enum import Enum
import json
import random
import transformers
import torch
import gc
import embedlib
import classify
import classify_sentiments
import time
import nltk
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

MAX_DIALOG_LEN = 10

class UserState:
    main_menu = 1
    dialog = 2

client = pymongo.MongoClient('localhost', 27017)
db = client['metachallenge']['users']
device = torch.device('cpu') if torch.cuda.device_count() == 0 else torch.device('cuda:0')
model = embedlib.models.GPT2Like('ru')
tokenizer = model.tokenizer
del model
model = transformers.GPT2LMHeadModel.from_pretrained(f'{embedlib.models.LIBPATH}/ru-GPT2Like')
model.to(device)
gc.collect()
max_add_length = 70
MAX_HIST_LENGTH = 500 - max_add_length
EOS_ID = 50000

def make_response(hist):
    input_ids = []
    for el in hist[::-1]:
        input_ids += [EOS_ID] + tokenizer.encode(el[0])[::-1]
    input_ids = input_ids[:MAX_HIST_LENGTH]
    input_ids = input_ids[::-1]
    start_len = len(input_ids)
    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
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

def cohesion_metric(dialog):
    return 1.0

positiveness_detector = classify_sentiments.Bert4SentimentClassification.load_from('positiveness_detector')
positiveness_detector.cpu()

def positiveness_metric(dialog):
    penalty = 0
    total = 0
    num = 0
    print('positiveness')
    for replic, actor in dialog:
        if actor == 1:
            num += 1
            result = positiveness_detector([replic]).view(-1).item()
            # neutral negative positive
            if result[1] >= 0.5:
                penalty += 0.1
            elif result[2] >= 0.5:
                penalty -= 1
            print(replic, result)
    print('penalty', penalty)
    if num != 0:
        return min(1.0, max(0.5 - penalty, 0))
    else:
        return 0

    return 1.0

insult_detector = classify.Bert4Classification.load_from('insult_detector')
insult_detector.cpu()

obscenity_words = json.load(open('ru_obscenity_dataset/obscenity_words.json'))
obscenity_words +=json.load(open('ru_obscenity_dataset/obscenity_words_extended.json'))
polite_words = [morph.parse(word)[0].normal_form for word in json.load(open('polite_words.json'))]

def politeness_metric(dialog):
    penalty = 0
    total = 0
    num = 0
    print('politeness')
    for replic, actor in dialog:
        if actor == 1:
            num += 1
            result = insult_detector([replic]).view(-1).item()
            if result <= .1:
                penalty += .1
            words = nltk.word_tokenize(replic)
            for word in words:
                infinitive = morph.parse(word)[:3]
                for el in infinitive:
                    if el.normal_form in obscenity_words:
                        penalty += .3
                    elif el.normal_form in polite_words:
                        penalty -= .05
            print(replic, result)
    print('penalty', penalty)
    if num != 0:
        return min(1.0, max(0.75 - penalty, 0))
    else:
        return 0

def from_score_to_mark(score):
    return score * 3 + 2

def get_score(dialog):
    cohesion = cohesion_metric(dialog)
    positiveness = positiveness_metric(dialog)
    politeness = politeness_metric(dialog)
    marks = [from_score_to_mark(el) for el in [cohesion, positiveness, politeness]]
    print('score', [cohesion, positiveness, politeness])
    print('marks', marks)
    return from_score_to_mark((cohesion + positiveness + politeness) / 3), marks

def init_user(key):
    db.insert_one({'id': key, 'last': [], 'dialogs': [], 'state': UserState.main_menu})

def update_record(user):
    db.update_one({'id': user['id']}, {'$set': user})

def stop_command(key):
    user = db.find_one({'id': key})
    if user is None:
        init_user(key)
        user = db.find_one({'id': key})
    user['state'] = UserState.main_menu
    if user['last'] == []:
        return 'Не найдено незавершенных диалогов'
    total, metrics = get_score(user['last'])

    user['last'] = []
    text = f"Вы прошли тест на {total:5.2f} баллов из 5.\n" \
           f"Ваши баллы за связность: {metrics[0]:5.2f}\n" \
           f"Ваши баллы за позитивность: {metrics[1]:5.2f}\n" \
           f"Ваши баллы за вежливость: {metrics[2]:5.2f}\n"
    user['last'].append((text, -1))
    user['dialogs'].append(user['last'])
    user['last'] = []
    update_record(user)
    return text

def eraseme_command(key):
    return db.remove({'id': key})

start_phrases = json.load(open('start_phrases.json'))['data']

def new_command(key):
    user = db.find_one({'id': key})
    user['state'] = UserState.dialog
    user['last'] = [(start_phrases[random.randint(0, len(start_phrases) - 1)], 0)]
    update_record(user)
    return user['last'][0][0]

commands = {
    '/stop': stop_command,
    '/eraseme': eraseme_command,
    '/new': new_command
}

def reply_to(key):
    user = db.find_one({'id': key})
    replic = make_response(user['last'])
    user['last'].append((replic, 0))
    update_record(user)
    return replic

def process_new_replic(key, replic):
    if replic is None:
        return
    replic = replic.strip()
    user = db.find_one({'id': key})
    if user is None:
        init_user(key)
    if replic in commands:
        return commands[replic](key)
    user = db.find_one({'id': key})

    if user['state'] == UserState.main_menu:
        return 'Чтобы начать новый диалог, напиши /new'
    else:
        user['last'].append((replic, 1))
        update_record(user)
        replic = reply_to(key)
        if len(user['last']) + 1 > MAX_DIALOG_LEN:
            replic = replic + '\n' + stop_command(key)
        return replic

if __name__ == '__main__':
    key = 'console'
    while True:
        response = input("Вы>> ")
        print(process_new_replic(key, response))
