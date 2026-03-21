import json

def load_data():
    dataDic = None
    with open("data/movies.json", "r", encoding="utf-8") as f:
        dataDic = json.load(f)
    return dataDic["movies"]

def load_stopwords():
    with open("data/stopwords.txt", "r") as f:
        return {line.strip() for line in f if line.strip()}