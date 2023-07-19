import tiktoken
from typing import List, Union, Dict
from HandleExceptions import CollatedExceptions
from PromptLLM import PromptLLM

class QaGeneration():
    """
    Object that contains methods for question, answer and summary generation
    """

    def __init__(
        self,
        prompt_llm: PromptLLM,
        collated_exceptions: CollatedExceptions,
        replace: bool = False
    ) -> None:
        """
        Constructor for QA generation

        Args:
            prompt_llm (PromptLLM): object to prompt the language model
            collated_exceptions (CollatedExceptions): object to help collate exceptions
            replace (bool, optional): replace tells the object to replace everything from the starting dataset. Defaults to False.
        """
        self.prompt_llm = prompt_llm
        self.collated_exceptions = collated_exceptions

        self.replace = replace

    def question_generation(
        self,
        definition: str,
        context_data: Dict
    ) -> List[Dict]:
        """
        Generating questions by prompting the model

        Args:
            definition (str): Definition for generating questions
            context_data (Dict): A Dict that contains the context

        Returns:
            List[Dict]: Returns a list of Dict containing the definition, context and the question
        """
        handle_exceptions = self.collated_exceptions.new_handle_exception(
            result_key="question",
            action="question",
            model_name=self.prompt_llm.current_model_name()
        )

        dataset: List[Dict] = []
        context = ensure_string(context_data["content"], "")
        try:
            output_questions = self.prompt_llm.prompt_model(
                definition, context, temp=0, max_tokens=150)
            question_list = output_questions.strip().splitlines()

            filter_words = ["umbrella", "discussion"]
            question_list = [
                item.strip() for item in question_list
                if item.strip() != "" and not self.check_present(filter_words, item.strip().lower())
            ]
            for question in question_list:
                result = {
                    "definition": definition,
                    "context": context,
                    "question": question
                }
                dataset.append(result)
        except Exception as e:
            exception_context = {
                "definition": definition,
                "context": context,
            }
            handle_exceptions.store_exceptions(exception_context, str(e))
        return dataset

    def answer_generation(
        self,
        definition: str,
        max_tokens: int,
        source_key: str,
        result_key: str,
        dataset: Dict,
        context_key: str = "",
        temp: int = 1
    ) -> Dict:
        """
        Prompts model to generate answers based on the definition and question

        Args:
            definition (str): Defines how the answers should be generated
            max_tokens (int): max tokens to be generated
            source_key (str): key for accessing the a source from the dict
            result_key (str): key within the dict to store the results
            dataset (Dict): dataset for the method to work with
            context_key (str, optional): key to access the context within the dict. Defaults to "".
            temp (int, optional): temperature for the model. Defaults to 1.

        Returns:
            Dict: dict containing the generated result and everything else the model uses
        """
        if not self.replace and result_key in dataset:
            return dataset

        handle_exceptions = self.collated_exceptions.new_handle_exception(
            result_key=result_key,
            action="answer",
            model_name=self.prompt_llm.current_model_name()
        )
        if source_key not in dataset:
            print(f"{source_key} cannot be found in the data_set")
            return dataset

        source: str = ensure_string(dataset[source_key], "")
        if (context_key != "" and context_key in dataset):
            context = ensure_string(dataset[context_key], "")
            source = f"context: {context} question: {source}"
        try:
            result = self.prompt_llm.prompt_model(
                definition, source, temp=temp, max_tokens=max_tokens)
            dataset[result_key] = result

        except Exception as e:
            exception_content = {
                "definition": definition,
                "source": source,
            }
            handle_exceptions.store_exceptions(exception_content, str(e))

            # Nothing generated
            dataset[result_key] = ""

        finally:
            return dataset

    def summarisation_generation(
        self,
        definition: str,
        max_tokens: int,
        source_key: str,
        result_key: str,
        intended_input_tokens: int,
        dataset: Dict,
        temp: int = 0
    ) -> Dict:
        """
        Convert text to summaries depending on the definition given

        Args:
            definition (str): Defines how summaries are to be generated
            max_tokens (int): Max tokens to generation
            source_key (str): Key within the dataset to access the source
            result_key (str): Key to store results in the dataset
            intended_input_tokens (int): Tokens of the source to be summarised
            dataset (Dict): Dataset with all information
            temp (int, optional): Temperature for the model. Defaults to 0.

        Returns:
            Dict: dataset with everything generated
        """
        if not self.replace and result_key in dataset:
            return dataset

        handle_exceptions = self.collated_exceptions.new_handle_exception(
            result_key=result_key,
            action="summarisation",
            model_name=self.prompt_llm.current_model_name()
        )
        source: str = ensure_string(dataset[source_key], "")

        try:
            # Check number of tokens
            encoding = tiktoken.get_encoding("r50k_base")
            if len(encoding.encode(source)) > intended_input_tokens:
                source_list: List = self.split_chunks(
                    source, intended_input_tokens, encoding)
            else:
                source_list: List = list(source)

            result = ""
            for source in source_list:
                result += self.prompt_llm.prompt_model(
                    definition, source, temp=temp, max_tokens=max_tokens)

            dataset[result_key] = result

        except Exception as e:
            exception_content = {
                "definition": definition,
                "source": source,
            }
            handle_exceptions.store_exceptions(exception_content, str(e))

            # Nothing generated
            dataset[result_key] = ""

        finally:
            return dataset

    @staticmethod
    def verify_key(key: str, dataset: Dict) -> bool:
        """
        Verifies if a key is within the datset

        Args:
            key (str): key in the dict
            dataset (Dict): dict dataset to check

        Returns:
            bool: if key is within dataset, True, else False
        """
        if key not in dataset:
            print(f"{key} not in dataset")
            return False
        return True

    @staticmethod
    def split_summaries(source: str | List) -> List[str]:
        """
        Split summaries by '\n'

        Args:
            source (str | List): summary to be splitted

        Returns:
            List[str]: list representation of the splitted summary
        """
        source = ensure_List_string(source)
        output_list = []
        for item in source:
            output_list.append(str(item).splitlines())
        return output_list

    @staticmethod
    def split_chunks(source: str | List, intended_input_tokens: int, encoding: tiktoken.Encoding) -> List[str]:
        """
        split text by tokens

        Args:
            source (str | List): text in string or list format
            intended_input_tokens (int): intended tokens for the summary chunks to be within
            encoding (tiktoken.Encoding): encoding for calculating tokens

        Returns:
            List[str]: list representation of summaries splitted into chunks
        """
        output_List = []
        current_string = ""
        source = ensure_List_string(source)
        for item in source:
            if len(encoding.encode(current_string + item)) < intended_input_tokens / 2:
                current_string += item
            else:
                output_List.append(current_string)
                current_string = ""
        return output_List

    @staticmethod
    def check_present(item_list: List, target: str) -> bool:
        """
        Checks if an item is present in a list
        """
        for item in item_list:
            if item in target:
                return True
        return False


"""
Functions out of class
"""


def ensure_string(text: Union[str, List[str]], joiner: str = " ") -> str:
    """
    Converts a list of string into a string, based on the joiner

    Args:
        text (Union[str, List[str]]): text to work on
        joiner (str, optional): how to join the list of string. Defaults to " ".

    Returns:
        str: string representation
    """
    if isinstance(text, List) and all(isinstance(item, str) for item in text):
        text = joiner.join(text)
    return str(text)


def ensure_List_string(text: Union[str, List[str]], separator: str = ". ") -> List[str]:
    """
    Splitting a string into a list of string, use sent_tokenize if splitting into sentences
    https://www.nltk.org/api/nltk.tokenize.html

    Args:
        text (Union[str, List[str]]): text to work on
        separator (str, optional): how to separate the string. Defaults to ". ".

    Returns:
        List[str]: list of string after separation
    """
    if isinstance(text, str):
        text = [item.strip() for item in text.split(separator)]
    return text
