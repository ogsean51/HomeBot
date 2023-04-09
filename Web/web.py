import requests
import json

def post():
    url = ""
    headers = {"Content-Type": "application/json; charset=utf-8"}
    json = json.stringify()
    resp = requests.post(url, headers=headers, data=json)
    print(resp)