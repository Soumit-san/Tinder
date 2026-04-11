import requests
with open("test.csv", "rb") as f:
    res = requests.post("http://127.0.0.1:8000/api/predict/batch", files={"file": f})
    print(res.status_code)
    print(res.text)
