import json
import jsonschema
from jsonschema import validate
import yaml


def load_config(path: str) -> dict:
    try:
        with open(path, "r") as stream:
            try:
                document = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                print("Yaml error:", e)
    except OSError as e:
        print("Read config error:", e)

    return document


def validate_config(obj: dict) -> bool:
    scheme_path = "./restrictions_schema.json"
    try:
        with open(scheme_path, "r") as file:
            schema = json.load(file)
    except OSError as e:
        print("Open file error:", e)
        return False
    except json.decoder.JSONDecodeError as e:
        print("Parse schema error:", e)
        return False

    try:
        validate(instance=obj, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        print("Validation error:", err)
        return False
    return True
