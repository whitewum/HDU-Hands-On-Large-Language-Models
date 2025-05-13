import requests
import json

url = "http://10.130.10.55:30000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}],
    "max_tokens": 100,
    "temperature": 0.7
}

response = requests.post(url, headers=headers, json=data)

# 打印响应
print(response.json())
