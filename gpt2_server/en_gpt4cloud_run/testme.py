import requests

url = "https://en-gpt2-server-b7e3qu3u4a-de.a.run.app" # "http://localhost:8080/"
resp = requests.post(url, json={"history": ["Hi! We offer you working for us for free. If you'd pay us, the weekends may be discussed."]})
print(resp.text)
