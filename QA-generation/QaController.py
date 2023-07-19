import json
import os
import random
from typing import Dict, List, Any

import tqdm
from evaluation import Evaluation
from QaGeneration import QaGeneration
from HandleExceptions import CollatedExceptions
from PromptLLM import PromptLLM


class QaController():
    def __init__(
        self,
        qa_config: Dict[Any, Any],
        model_name: str,
        num_of_generations: int,
        starting_dataset_path: str = "",
        context_name: str = "",
        questions_path: str = "",
        replace: bool = False,
        identifier: str = ""
    ) -> None:
        """
        Constructor the QA controller

        Args:
            qa_config (Dict[Any, Any]): qa config path
            model_name (str): name of model to use, should align with qa-config
            num_of_generations (int): number of intended generations
            starting_dataset_path (str, optional): path to a dataset to continue generations from. Defaults to "".
            context_name (str, optional): context name, should align with the directory name. Defaults to "".
            questions_path (str, optional): Questions path to use with answer generation. Defaults to "".
            replace (bool, optional): When True, new generations will replace old ones in starting dataset. Defaults to False.
            identifier (str, optional): Unique identifier for the generated files. Defaults to "".
        """
        # CONFIGS AND ARGS
        self.qa_config = qa_config

        self.identifier = identifier

        # GENERATION ARGS
        self.num_of_generations = num_of_generations

        self.context_name = context_name
        self.generation_file_path = f"{self.qa_config['file_config']['generation_dir']}/{self.context_name}"

        # GETTING CONFIGS
        with open(self.qa_config['file_config']['definition_path'], "r") as f:
            self.definition_data = json.load(f)

        # LOADING QUESTIONS
        self.questions_path = questions_path
        if questions_path != "" and os.path.exists(questions_path):
            with open(questions_path, "r") as f:
                self.questions_dataset = json.load(f)
        else:
            self.questions_dataset = []
            print("no questions loaded")

        # LOADING STARTING DATASET
        if starting_dataset_path != "" and os.path.exists(starting_dataset_path):
            with open(starting_dataset_path, "r") as f:
                self.starting_dataset = json.load(f)
        else:
            self.starting_dataset = []
            print("no starting dataset loaded")

        # EXCEPTIONS
        self.collated_exceptions = CollatedExceptions(
            qa_config['file_config']['logs_dir'])

        # LLM
        self.prompt_llm = PromptLLM(
            model_name=model_name,
            qa_config=self.qa_config
        )

        # EVALUATOR
        self.evaluation_object = Evaluation(
            collated_exceptions=self.collated_exceptions
        )

        # QA GENERATOR
        self.qa_object = QaGeneration(
            prompt_llm=self.prompt_llm,
            collated_exceptions=self.collated_exceptions,
            replace=replace
        )

    def open_book_qa(
        self
    ) -> None:
        """
        Performs open book QA. Context will be given with questions during answer generation
        """
        # Checking questions dataset
        if len(self.questions_dataset) <= 0:
            print("questions not loaded correctly")
            exit()

        # Progress bar
        progress_bar = tqdm.tqdm(
            total=min(self.num_of_generations, len(self.questions_dataset)),
            desc=f"{self.context_name}, {self.prompt_llm.current_model_name()}, {self.identifier}"
        )
        target_dataset = self.starting_dataset
        progress_bar.update(len(target_dataset))

        # Does all the steps per index
        for idx in range(len(target_dataset), min(self.num_of_generations, len(self.questions_dataset))):

            # reference to data to work with
            working_dataset = {
                "context": self.questions_dataset[idx]["context"],
                "question": self.questions_dataset[idx]["question"],
            }

            if "concise_context" in self.questions_dataset[idx] and self.questions_dataset[idx]["concise_context"] != "":
                working_dataset["concise_context"] = self.questions_dataset[idx]["concise_context"]
            else:
                progress_bar.set_postfix({'Info': "creating concised context"})
                # Summarise the context
                working_dataset = self.qa_object.summarisation_generation(
                    definition=self.definition_data["summarise_to_text"],
                    max_tokens=1024,
                    source_key="context",
                    result_key="concise_context",
                    intended_input_tokens=1024,
                    dataset=working_dataset
                )
                self.questions_dataset[idx]["concise_context"] = working_dataset["concise_context"]
                with open(self.questions_path, "w") as f:
                    json.dump(self.questions_dataset, f, indent=2)

            # Generates answer from the summarised context
            progress_bar.set_postfix(
                {'Info': "answering with concised context"})
            working_dataset = self.qa_object.answer_generation(
                definition=self.definition_data["answer_with_context"],
                max_tokens=1024,
                source_key="question",
                context_key="concise_context",
                result_key="open_book_answer",
                dataset=working_dataset
            )
            # Evaluates the raw answer against the raw context
            progress_bar.set_postfix({'Info': "evaluating concised answer"})
            working_dataset = self.evaluation_object.evaluation_generation(
                dataset=working_dataset,
                cand_key="open_book_answer",
                ref_key="concise_context",
                result_key="open_book_orignals",
            )
            # Updates the dataset
            target_dataset.append(working_dataset)

            # Saving exceptions
            self.collated_exceptions.save_failures()

            # save for every iteration
            with open(f"{self.generation_file_path}/open_book_answers_{self.context_name}_{self.identifier}_{self.prompt_llm.get_chat_model()}.json", 'w') as f:
                json.dump(target_dataset, f, indent=2)

            progress_bar.update(1)

    def close_book_qa(self) -> None:
        """
        Performs blind QA, where only questions will be given to the model for answer generation
        """
        # Checking questions dataset
        if len(self.questions_dataset) <= 0:
            print("questions not loaded correctly")
            exit()

        target_dataset = self.starting_dataset
        progress_bar = tqdm.tqdm(
            total=min(self.num_of_generations, len(self.questions_dataset)),
            desc=f"{self.context_name}, {self.prompt_llm.current_model_name()}, {self.identifier}"
        )
        progress_bar.update(len(target_dataset))
        for idx in range(len(target_dataset), min(self.num_of_generations, len(self.questions_dataset))):

            # reference to data to work with
            working_dataset = {
                "context": self.questions_dataset[idx]["context"],
                "question": self.questions_dataset[idx]["question"],
            }

            progress_bar.set_postfix({'Info': "generating answers"})
            # Generates answer from the questions
            working_dataset = self.qa_object.answer_generation(
                definition=self.definition_data["answer"],
                max_tokens=1024,
                source_key="question",
                result_key="close_book_answer",
                dataset=working_dataset
            )
            # Generates point form of the answer
            progress_bar.set_postfix({'Info': "summarising answers"})
            working_dataset = self.qa_object.answer_generation(
                definition=self.definition_data["summarise_to_points"],
                temp=0,
                max_tokens=250,
                source_key="close_book_answer",
                result_key="point_form_close_book_answer",
                dataset=working_dataset
            )
            if "point_form_context" in self.questions_dataset[idx] and self.questions_dataset[idx]["point_form_context"] != "":
                working_dataset["point_form_context"] = self.questions_dataset[idx]["point_form_context"]
            else:
                # Generates point form of the context
                progress_bar.set_postfix({'Info': "summarising context"})
                working_dataset = self.qa_object.answer_generation(
                    definition=self.definition_data["summarise_to_points"],
                    temp=0,
                    max_tokens=250,
                    source_key="context",
                    result_key="point_form_context",
                    dataset=working_dataset
                )
                self.questions_dataset[idx]["point_form_context"] = working_dataset["point_form_context"]
                with open(self.questions_path, "w") as f:
                    json.dump(self.questions_dataset, f, indent=2)

            # Evaluates the raw answer against the raw context
            progress_bar.set_postfix({'Info': "evaluating answers"})
            working_dataset = self.evaluation_object.evaluation_generation(
                dataset=working_dataset,
                cand_key="close_book_answer",
                ref_key="context",
                result_key="answer",
            )
            # Evaluates the point form version of answer to the point form version of context
            progress_bar.set_postfix({'Info': "evaluating summarised answers"})
            working_dataset = self.evaluation_object.evaluation_generation(
                dataset=working_dataset,
                cand_key="point_form_close_book_answer",
                ref_key="point_form_context",
                result_key="summarised",
            )
            # Updates the dataset
            target_dataset.append(working_dataset)

            # Saving exceptions
            self.collated_exceptions.save_failures()

            # save for every iteration
            with open(f"data_2021_nyt.json", 'w') as f:
                json.dump(target_dataset, f, indent=2)

            progress_bar.update(1)

    def generate_questions(self, context_file_name: str) -> None:
        """
        Generates questions for the target dataset.

        Args:
            context_file_name (str): file path for context to generate questions from
        """
        context_file_path = f"{self.qa_config['file_config']['context_dir']}/{self.context_name}"
        generation_file_path = f"{self.qa_config['file_config']['generation_dir']}/{self.context_name}"
        with open(f"{context_file_path}/{context_file_name}.json", "r") as f:
            context_json = json.load(f)

        target_dataset: List[Dict[str, str]] = self.questions_dataset

        progress_bar = tqdm.tqdm(
            total=self.num_of_generations,
            desc=f"question generation: {self.prompt_llm.current_model_name()}"
        )
        progress_bar.update(len(target_dataset))

        # Filling up the dataset to hit number of generations target
        while (len(target_dataset) < self.num_of_generations):
            random_number = random.randint(0, len(context_json) - 1)
            context_data = context_json[random_number]
            result_list = self.qa_object.question_generation(
                definition=self.definition_data["question"],
                context_data=context_data
            )
            target_dataset += result_list
            progress_bar.update(len(result_list))

        with open(f"{generation_file_path}/questions_{self.context_name}_{context_file_name}_{self.prompt_llm.get_chat_model()}.json", 'w') as f:
            json.dump(target_dataset, f, indent=2)
