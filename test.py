import time
import requests

url = "http://0.0.0.0:8000/inference"
data = {"text": "Скидка предоставляется 20 процентов"}

start = time.time()
response = requests.get(url, json=data)
finish = time.time()

print(f'Time taken: {finish - start:.2f}s\n')
print(response.json())
