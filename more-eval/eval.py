import json
from QaGeneration import ensure_string, ensure_List_string
from evaluation import Evaluation
from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import cast, List, Dict
import tqdm
from nltk.tokenize import sent_tokenize

"""
This file is used for calculating bertScore, sentence-bert and rouge scores seperately
"""

filename_list = [
    "../data/generations/nyt/close_book_answers_nyt_2021_batch_vicuna-13b-v1.3.json",
    "../data/generations/nyt/open_book_answers_nyt_vicuna-13b-v1.3.json",
    "../data/generations/rsis/close_book_answers_rsis_2021_batch_vicuna-13b-v1.3.json",
    "../data/generations/rsis/close_book_answers_rsis_vicuna-13b-v1.3.json",
    "../data/generations/straitstimes/close_book_answers_straitstimes_vicuna-13b-v1.3.json"
]

def clean_data(answer_dataset:Dict) -> Dict:
    """
    Ensures that the context is a string

    Args:
        answer_dataset (Dict): the dict representing 1 generation instance

    Returns:
        Dict: returns the updated dataset
    """
    answer_dataset["context"] = ensure_string(answer_dataset["context"], joiner=" ")

    return answer_dataset

def eval(answer_dataset:Dict, context_key:str, answer_key:str, filename:str) -> Dict:
    """
    Performs evaluations using bertScore, sentence-bert and rouge

    Args:
        answer_dataset (Dict): dataset to reference from
        context_key (str): key for the context, which acts as the reference
        answer_key (str): Key for the answer, which acts as the candidate
        filename (str): filename to work on

    Returns:
        Dict: Returns the updated dataset
    """
    context = ensure_string(answer_dataset[context_key], joiner=" ")
    answer_str = ensure_string(answer_dataset[answer_key])
    answer_dataset[context_key] = sent_tokenize(context)

    embedder = SentenceTransformer('all-mpnet-base-v2')
    answer = Evaluation.process_answer(answer_dataset[answer_key])

    st_eval_list = Evaluation.eval_sentence_transformer(embedder=embedder, ref_window=answer_dataset[context_key], cand_window=answer)
    overall_list = Evaluation.eval_sentence_transformer_str(embedder=embedder, ref_window=answer_str, cand_window=context)

    answer_dataset = Evaluation.log_eval_score(filename, answer_dataset, st_eval_list, is_blank=False)
    answer_dataset["overall_answer_cosine"] = overall_list

    return answer_dataset

if __name__=="__main__":
    new_answer_dataset = []
    for filename in filename_list:
        with open(filename, "r") as f:
            answer_dataset_list = json.load(f)

        progress_bar = tqdm.tqdm(
            total=len(answer_dataset_list), 
            desc=f"{filename}"
        )

        for answer_dataset in answer_dataset_list:

            answer_dataset = eval(
                answer_dataset=answer_dataset,
                context_key="context",
                answer_key="close_book_answer",
                filename="answer_sentence_transformer"
            )

            with open(filename, "w") as f:
                json.dump(answer_dataset_list, f, indent=2)

            answer_dataset = eval(
                answer_dataset=answer_dataset,
                context_key="point_form_context",
                answer_key="point_form_close_book_answer",
                filename="summarised_sentence_transformer"
            )

            with open(filename, "w") as f:
                json.dump(answer_dataset_list, f, indent=2)

            progress_bar.update(1)

            new_answer_dataset.append(answer_dataset)
            with open("data.json", "w") as f:
                json.dump(new_answer_dataset, f, indent=2)
