import requests
import json
API_URL = "http://localhost:3000/api/predict"
payload = {
    "api_key": "GF5DI92953LI10M",
    "text": "Продаю новую страйкбольную винтовку ASG и тактический жилет. В комплекте магазин и ремень. Состояние идеальное.",
    "photo_urls": [
        "https://images.unsplash.com/photo-1734000403582-da52e3699c0c?q=80&w=686&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1734555772511-324448b754df?q=80&w=699&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    ]
}

try:
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")