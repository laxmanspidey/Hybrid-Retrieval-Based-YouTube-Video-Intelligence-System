import requests, json

data = {
    'model': 'qwen3.5:4b',
    'messages': [
        {'role': 'system', 'content': 'You are a helpful assistant. Context: The VSS is pure stealth.'},
        {'role': 'user', 'content': 'which weapon is pure stealth'},
        {'role': 'assistant', 'content': 'The VSS is pure stealth.'},
        {'role': 'user', 'content': 'can you tell me more about it'}
    ],
    'stream': False
}

try:
    resp = requests.post('http://localhost:11434/api/chat', json=data)
    with open('api_resp.json', 'w', encoding='utf-8') as f:
        f.write(resp.text)
except Exception as e:
    with open('api_resp.json', 'w', encoding='utf-8') as f:
        f.write(str(e))
