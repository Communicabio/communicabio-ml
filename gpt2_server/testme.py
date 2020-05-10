import requests
import pickle
import json
resp = requests.post('http://localhost:8000', json={'history': ['Привет! как дела?',
                                                'Привет! Нормально. Чем займемся?']})
resp.raise_for_status()
text = json.loads(resp.text)['text']
print(text)
