import requests
import pickle
import json
# 'https://gpt2-service-b7e3qu3u4a-lz.a.run.app' http://localhost:8080
resp = requests.post('https://gpt2-service-b7e3qu3u4a-lz.a.run.app', json={'history': ['Привет! как дела?',
                                                'Привет! Нормально. Чем займемся?']})
resp.raise_for_status()
text = json.loads(resp.text)['text']
print(text)
