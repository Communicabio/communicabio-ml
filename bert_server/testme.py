import requests
import pickle
import json
resp = requests.post('https://bert-service-b7e3qu3u4a-ew.a.run.app', json={'text': 'Привет, мир!'})
resp.raise_for_status()
tensor = json.loads(resp.text)['vector']
print(tensor)
