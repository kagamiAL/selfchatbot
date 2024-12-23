import os
import importlib
from .preprocess_data import add_preprocessor

preprocessors_directory = os.path.join(os.path.dirname(__file__), "Preprocessors")

# * May or may not change this later
for file in os.listdir(preprocessors_directory):
    if file.endswith(".py"):
        module_name = file[:-3]
        add_preprocessor(
            module_name,
            importlib.import_module(
                "DataProcessing.Preprocessors." + module_name
            ).preprocess,
        )
