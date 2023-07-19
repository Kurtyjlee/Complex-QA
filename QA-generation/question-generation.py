import argparse
import sys
import yaml
from QaController import QaController

"""
usage: 
python3 question-generation.py \
    --context_name str|list[str] \
    --num_of_generations int \
    --model_name str|list[str] \
    --qa_config_path str \
    --questions_path str (optional)

Example: 
python3 question-generation.py 
    --context_name rsis \
    --num_of_generations 1 \
    --model_name vicuna-13b-v1.3 \
    --qa_config_path ./configs/QA_config.yaml \
    --questions_path ../data/generations/straitstimes/questions_straitstimes_vicuna-13b-v1.3.json
"""


def parse_args():
    """
    Parse args configurations for the script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_name",
        type=str,
        required=True,
        default="rsis",
        help="The directory name where context data(s) is/are stored, 'rsis,nyt,straitstimes",
    )
    parser.add_argument(
        "--num_of_generations",
        type=int,
        default=1,
        help="number of generations to generate questions from",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vicuna-13b-v1.3",
        help="model name(s), multiple models separated by commas 'vicuna,gpt-3.5'",
    )
    parser.add_argument(
        "--qa_config_path",
        type=str,
        default="/home/lyijie/self-instruct/code/configs/QA_config.yaml",
        help="config for the model",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default="",
        help="Full path for the starting dataset",
    )
    parser.add_argument(
        "--context_file_name",
        type=str,
        required=True,
        help="name of file to get context from, without the .json",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    args.context_name = [item.strip() for item in args.context_name.split(",")]
    args.model_name = [item.strip() for item in args.model_name.split(",")]

    # Getting the config dict
    try:
        with open(args.qa_config_path, "r") as f:
            qa_config = yaml.safe_load(f)
    except Exception as e:
        print(str(e))
        print("--qa_config only takes in a yaml config file")
        sys.exit()

    # Allow for iterations through models and contexts
    for model_name in args.model_name:
        for context_name in args.context_name:
            qa_controller = QaController(
                qa_config=qa_config,
                model_name=model_name,
                questions_path=args.questions_path,
                num_of_generations=args.num_of_generations,
                context_name=context_name,
            )
            qa_controller.generate_questions(
                context_file_name=args.context_file_name)
