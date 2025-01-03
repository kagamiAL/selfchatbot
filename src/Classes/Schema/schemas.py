# The preprocess schema for preprocessing parameters
preprocess_parameters_schema = {
    "$id": "Schema/PreprocessParameters.json",
    "type": "object",
    "properties": {
        "model": {"type": "string"},
        "type_fine_tune": {"type": "string"},
        "max_length": {"type": "integer", "default": 1024},
        "preprocessor_data": {
            "type": "object",
            "default": {},
            "properties": {},
            "additionalProperties": True,
        },
    },
    "required": ["model", "type_fine_tune"],
    "additionalProperties": False,
}
