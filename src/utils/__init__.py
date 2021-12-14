import json
import jsonschema
import yaml


def load_config(path: str) -> dict:
    with open(path, 'r') as stream:
        document = yaml.safe_load(stream)
    return document


def validate_config(obj: dict):
    scheme_path = './restrictions_schema.json'
    with open(scheme_path, 'r') as file:
        schema = json.load(file)
    jsonschema.validate(instance=obj, schema=schema)


def get_config(path: str) -> dict:
    path_to_default_config = './default_config.yaml'

    custom_config = load_config(path)
    default_config = load_config(path_to_default_config)

    validate_config(custom_config)
    validate_config(default_config)
    default_config.update(custom_config)
    return default_config
