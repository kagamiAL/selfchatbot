import json
from pathlib import Path
from jsonschema import validate, ValidationError, Draft7Validator


def apply_defaults(data, schema):
    """
    Recursively applies default values to a given data according to a schema. If a key is missing from the data, and the schema
    specifies a default value, the default value is inserted into the data. If a key points to a nested object, this function
    is called recursively on that object.

    After filling in default values, the data is validated against the schema to ensure that all required keys are present and
    that all values conform to the schema.

    Args:
        data (dict): The data to fill in defaults for
        schema (dict): The schema to use for filling in defaults

    Returns:
        dict: The modified data with default values filled in
    """
    validator = Draft7Validator(schema)
    for prop, subschema in schema.get("properties", {}).items():
        if "default" in subschema and prop not in data:
            data[prop] = subschema["default"]
        elif subschema.get("type") == "object" and isinstance(data.get(prop), dict):
            apply_defaults(data[prop], subschema)
    validator.validate(data)
    return data


def validate_dict(data, schema, resolver=None):
    """
    Validates a given dictionary against a given schema.

    Args:
        data (dict): The dictionary to validate
        schema (dict): The schema to validate against

    Raises:
        ValueError: If the dictionary does not conform to the schema
    """
    try:
        validate(instance=data, schema=schema, resolver=resolver)
    except ValidationError as e:
        raise e


def get_validated_json(file_path: Path | str, schema: dict) -> dict:
    """
    Validates a JSON file against a given schema.

    Args:
        file_path (Path | str): The path to the JSON file to validate
        schema (dict): The schema to validate against

    Returns:
        dict: The validated JSON data

    Raises:
        ValueError: If the JSON file is invalid or does not conform to the schema
    """

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        validate_dict(data, schema)
        return apply_defaults(data, schema)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")
    except FileNotFoundError:
        raise ValueError("File not found.")
