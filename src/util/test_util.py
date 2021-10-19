import json
import os
import pytest

from . import load_config, validate_config


@pytest.fixture(scope='function')
def resource_setup():
    print('\n Init')
    examples_path = './config_examples'
    examples = next(os.walk(examples_path), (None, None, []))[2]

    return [{'name': ex, 'config': load_config(os.path.join(examples_path, ex))} for ex in examples]


def test_validate_examples(resource_setup):
    for config in resource_setup:
        # print(json.dumps(config, indent=2))
        validate(config)


def validate(obj):
    print('\n', obj['name'])
    if obj['name'].split('_')[0] == 'good':
        assert validate_config(obj['config'])
    else:
        assert not validate_config(obj['config'])
