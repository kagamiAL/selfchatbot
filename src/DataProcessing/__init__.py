import pkgutil
import importlib
import inspect
from .preprocess_data import add_preprocessor
from Classes.Preprocessors.preprocessor import Preprocessor


def get_classes_from_module(module_name: str, parent_class):
    """
    Recursively get all classes in a module or its submodules that inherit from a specified parent class.

    Args:
        module_name (str): The name of the module to search.
        parent_class (class): The parent class to inherit from.

    Returns:
        dict: A dictionary containing the data format of the class as the key and the class object as the value.
    """
    discovered_classes = {}

    package = importlib.import_module(module_name)

    for _, submodule_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            submodule = importlib.import_module(submodule_name)

            for _, obj in inspect.getmembers(submodule, inspect.isclass):
                if (
                    obj.__module__ == submodule_name
                    and issubclass(obj, parent_class)
                    and obj is not parent_class
                ):
                    discovered_classes[obj.DATA_FORMAT] = obj
        except Exception as e:
            print(f"Error importing module {submodule_name}: {e}")

    return discovered_classes


for data_format, preprocessor in get_classes_from_module(
    "Classes.Preprocessors", Preprocessor
).items():
    print(f"Adding preprocessor for {data_format}")
    add_preprocessor(data_format, preprocessor)
