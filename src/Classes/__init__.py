import importlib
import inspect
import pkgutil
from Classes.Formatters.formatter import Formatter, register_formatter


def get_classes_from_module(module_name: str, parent_class) -> list:
    """
    Recursively get all classes in a module or its submodules that inherit from a specified parent class.

    Args:
        module_name (str): The name of the module to search.
        parent_class (class): The parent class to inherit from.

    Returns:
        list: A list of class objects that inherit from the specified parent class.
    """
    discovered_classes = []

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
                    discovered_classes.append(obj)
        except Exception as e:
            print(f"Error importing module {submodule_name}: {e}")

    return discovered_classes


def initialize_formatters():
    """
    Initializes all the formatters by registering them in the formatters dictionary
    """

    for formatter in get_classes_from_module("Classes.Formatters", Formatter):
        register_formatter(formatter.MODEL_NAME, formatter)


initialize_formatters()
