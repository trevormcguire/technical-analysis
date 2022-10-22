import json


def write_json(obj, filepath: str):
    with open(filepath, "w") as f:
        f.write(json.dumps(obj))

def load_json(filepath: str):
    with open(filepath, "r") as f:
        return json.loads(f)