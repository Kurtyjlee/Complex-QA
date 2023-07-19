import json
from typing import List, Dict
from datetime import datetime


class HandleExceptions():
    """
    Class to represent a single list of exceptions to be logged
    """
    def __init__(self):
        self.exceptions: List[Dict] = []
        self.number_of_fails: int = 0

    def __str__(self):
        """
        String representation to display number of fails
        """
        return f"number of fails: {self.number_of_fails}"

    def store_exceptions(self, content: Dict, exception: str) -> None:
        """
        stores an exception instance

        Args:
            content (Dict): descriptor for the exception
            exception (str): the exception message itself
        """
        content["error"] = exception

        self.exceptions.append(content)
        self.number_of_fails += 1

    def get_exceptions(self) -> List:
        """
        returns all stored exceptions
        """
        return self.exceptions

    def combine_num_of_fails(self) -> None:
        """
        add the number of fails to the front of the list
        """
        self.exceptions = [
            {"number of fails": self.number_of_fails}] + self.exceptions


class CollatedExceptions():
    """
    Collated list of HandleExceptions. HandleExceptions represent a list of exceptions for a single generation.
    """
    def __init__(self, directory: str):
        """
        Contructor for CollatedExceptions

        Args:
            directory (str): Directory to store the exceptions
        """
        self.file_path: str = self.generate_file_path(directory)
        self.collated_exceptions: Dict[str, HandleExceptions] = {}

    def generate_file_path(self, directory: str) -> str:
        """
        Generates a new file path for exceptions, with datetime included

        Args:
            directory (str): Directory to decorate

        Returns:
            str: File path generated
        """
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")

        return f"{directory}/error_log_{formatted_datetime}.json"

    def create_exception(self, name: str) -> HandleExceptions:
        """
        Gives a new HandleException, stored in the CollatedExceptions object. Ensure that the name is unique

        Args:
            name (str): Represents the key of the HandleException

        Returns:
            HandleExceptions: Returns the HandleException stored
        """
        if name not in self.collated_exceptions:
            self.collated_exceptions[name] = HandleExceptions()

        return self.collated_exceptions[name]

    def new_handle_exception(self, result_key: str, action: str, model_name: str) -> HandleExceptions:
        """
        Auto generate a new HandleException Object, based on the result_key, action and model name

        Args:
            result_key (str): result_key of generation
            action (str): what the generation is doing
            model_name (str): model used

        Returns:
            HandleExceptions: returned a stored handle exception
        """
        generation_name = f"{result_key}_{action}_{model_name}"
        handle_exceptions = self.create_exception(name=generation_name)

        return handle_exceptions

    def save_failures(self) -> None:
        """
        Saves the CollatedException to a json file
        """
        result: dict[str, List[Dict]] = {}
        for key, value in self.collated_exceptions.items():
            value.combine_num_of_fails()
            result[key] = value.get_exceptions()

        with open(self.file_path, "w") as f:
            json.dump(result, f, indent=2)
