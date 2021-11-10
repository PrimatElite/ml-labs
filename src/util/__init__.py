import json
import jsonschema
from jsonschema import validate
import yaml


def load_config(path: str) -> dict:
    try:
        with open(path, "r") as stream:
            document = yaml.safe_load(stream)
    except OSError as e:
        print("Read config error:", e)
        return {}
    except yaml.YAMLError as e:
        print("Yaml error:", e)
        return {}

    return document


def validate_config(obj: dict) -> bool:
    scheme_path = "./restrictions_schema.json"

    try:
        with open(scheme_path, "r") as file:
            schema = json.load(file)
        validate(instance=obj, schema=schema)
    except OSError as e:
        print("Open file error:", e)
        return False
    except json.decoder.JSONDecodeError as e:
        print("Parse schema error:", e)
        return False
    except jsonschema.exceptions.ValidationError as err:
        print("Validation error:", err)
        return False

    return True


def get_config(path: str) -> dict:
    path_to_default_config = "./default_config.yaml"

    custom_config = load_config(path)
    default_config = load_config(path_to_default_config)

    if validate_config(custom_config) and validate_config(default_config):
        default_config.update(custom_config)
        return default_config

    return {}
