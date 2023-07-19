import argparse
import sys
import yaml
from QaController import QaController

"""
usage: 
python3 open-book-generation.py \
    --context_name str|list[str] \
    --questions_path str \
    --num_of_generations int \
    --model_name str|list[str] \
    --qa_config str \
    --starting_dataset_path str (optional) \
    --starting_index int (optional) \
    --replace bool (optional) 

Example: 
python3 open-book-generation.py \
    --context_name rsis \
    --questions_path ../data/generations/rsis/questions_rsis_vicuna-13b-v1.3.json \
    --num_of_generations 1 \
    --model_name vicuna-13b-v1.3 \
    --qa_config ./configs/QA_config.yaml \
    --starting_dataset_path ../data/generations/rsis/open_book_answers_rsis_vicuna-13b-v1.3.json \
    --starting_index 0 \
    --replace True
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="The path(s) to find all the questions, multiple paths separated by commas, ensure that question path correspond with context",
    )
    parser.add_argument(
        "--context_name",
        type=str,
        required=True,
        default="rsis",
        help="name of the context, should align with the context directory",
    )
    parser.add_argument(
        "--num_of_generations",
        type=int,
        default=1,
        help="number of answers to generate",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vicuna-13b-v1.3",
        help="model name(s), multiple models separated by commas 'vicuna,gpt-3.5'",
    )
    parser.add_argument(
        "--qa_config",
        type=str,
        default="/home/lyijie/self-instruct/code/configs/QA_config.yaml",
        help="path to config for the model",
    )
    parser.add_argument(
        "--starting_dataset_path",
        type=str,
        default="",
        help="Path for the starting dataset, meant for continuation",
    )
    parser.add_argument(
        "--replace",
        type=bool,
        default=False,
        help="Whether to replace old generations within starting dataset",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        required=True,
        help="unqiue identifier for the generation file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    context_name_list = [item.strip()
                         for item in str(args.context_name).split(",")]
    model_name_list = [item.strip()
                       for item in str(args.model_name).split(",")]
    questions_path_list = [item.strip()
                           for item in str(args.questions_path).split(",")]

    # Getting the config dict
    try:
        with open(args.qa_config, "r") as f:
            qa_config = yaml.safe_load(f)
    except Exception as e:
        print(str(e))
        print("--qa_config only takes in a yaml config file")
        sys.exit()

    # Allow for iterations through models and contexts
    for model_name in model_name_list:
        for index, context_name in enumerate(context_name_list):
            qa_controller = QaController(
                qa_config=qa_config,
                model_name=model_name,
                num_of_generations=args.num_of_generations,
                starting_dataset_path=args.starting_dataset_path,
                context_name=context_name,
                questions_path=questions_path_list[index],
                replace=args.replace,
                identifier=args.identifier
            )
            qa_controller.open_book_qa()
