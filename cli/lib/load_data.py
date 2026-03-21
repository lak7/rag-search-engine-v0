import json

def load_data():
    dataDic = None
    with open("data/movies.json", "r", encoding="utf-8") as f:
        dataDic = json.load(f)
    return dataDic["movies"]