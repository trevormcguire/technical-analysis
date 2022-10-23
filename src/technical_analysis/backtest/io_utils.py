import json
import pickle


def save_json(obj, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(obj))

def load_json(filepath: str):
    with open(filepath, "r") as f:
        return json.loads(f)

def save_pickle(obj, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)
