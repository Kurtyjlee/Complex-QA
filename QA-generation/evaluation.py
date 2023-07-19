import argparse
import glob
import logging
from typing import cast
import transformers
from bert_score import BERTScorer
import json
from sentence_transformers import SentenceTransformer, util
import torch
from torch import Tensor
from rouge_score import rouge_scorer
import os
import nltk
import ssl
from HandleExceptions import CollatedExceptions
from QaGeneration import ensure_string
from typing import List, Dict
from nltk.tokenize import sent_tokenize

os.environ["NLTK_DATA"] = "~/nltk_data"
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

"""
This doc uses 2 evaluation methods, sentence bert and bert score
ref == reference paragraph
cand == candidate paragraph, compared against the reference paragraph
"""


class Evaluation():
    def __init__(self, collated_exceptions: CollatedExceptions):
        """
        Constructor for the Evaluation objext

        Args:
            collated_exceptions (CollatedExceptions): Object for logging exceptions
        """
        self.collated_exceptions = collated_exceptions

        transformers.tokenization_utils.logger.setLevel(logging.ERROR)
        transformers.configuration_utils.logger.setLevel(logging.ERROR)
        transformers.modeling_utils.logger.setLevel(logging.ERROR)

    @staticmethod
    def eval_bert_score(
        scorer: BERTScorer,
        cand_window: List,
        ref_window: List
    ) -> List:
        """
        Evaluations using BertScore, compares each string within the candidate to every
        string in the reference
        https://pypi.org/project/bert-score/

        Args:
            scorer (BERTScorer): Bert Scorer object
            cand_window (List): list of string for the candidate passage
            ref_window (List): list of string for the reference passage

        Returns:
            List: returns the evaluations done by Bert Score
        """
        bs_eval_list = []
        for cand in cand_window:
            P, R, F1 = scorer.score([cand], [ref_window])
            F1 = cast(Tensor, F1)
            bs_eval_list.append(F1.item())
        return bs_eval_list

    @staticmethod
    def eval_sentence_transformer(
        embedder: SentenceTransformer,
        cand_window: List,
        ref_window: List,
    ) -> List:
        """
        Evaluations using sentence-bert, compares each string within the candidate to every
        string in the reference
        https://www.sbert.net/

        Args:
            embedder (SentenceTransformer): embedder object for sentence bert
            cand_window (List): list of string for the candidate
            ref_window (List): list of string for the reference

        Returns:
            List: evaluations done by sentence-bert
        """
        top_k = 1
        st_eval_list = []

        # Embeddings for sentence-bert
        ref_embeddings = embedder.encode(ref_window, convert_to_tensor=True)
        ref_embeddings = cast(Tensor, ref_embeddings)

        for cand in cand_window:
            cand_embedding = embedder.encode(cand, convert_to_tensor=True)
            cand_embedding = cast(Tensor, cand_embedding)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(cand_embedding, ref_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            # Score is a tensor object
            for score, idx in zip(top_results[0], top_results[1]):
                st_eval_list.append(score.item())
        return st_eval_list

    @staticmethod
    def log_eval_score(
        eval_name: str,
        data: dict,
        eval_list: List,
        is_blank: bool = False
    ) -> dict:
        """
        Logger to log the evaluation scores in the given dict

        Args:
            eval_name (str): name of evaluation done
            data (dict): dataset to work on
            eval_list (List): evaluations done by the bert model
            is_blank (bool, optional): Whether the list given is blank, error has occurred. Defaults to False

        Returns:
            dict: data with the evaluations logged
        """
        if not is_blank and len(eval_list) != 0:
            average = float(sum(eval_list)) / len(eval_list)
            data[f"{eval_name}_spread"] = eval_list
            data[f"{eval_name}_average"] = average
        else:
            data[f"{eval_name}_spread"] = [0]
            data[f"{eval_name}_average"] = 0
        return data

    @staticmethod
    def save_to_json(
        source_path: str,
        filename: str,
        data: List[Dict],
        message: str = ""
    ) -> None:
        """
        Saves the dataset to json

        Args:
            source_path (str): directory for the file
            filename (str): filename to save the file
            data (List[Dict]): data to save
            message (str, optional): identifier for the file, Defaults to ""
        """
        file = filename.replace(".json", "")
        with open(f"{source_path}/{file}{message}.json", "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def process_answer(answer: str | List[str]) -> List[str]:
        """
        Splitting the answers into sentences

        Args:
            answer (str | List[str]): If answer is string, use a bert model to split into sentences

        Returns:
            List[str]: returns a list of splitted answers
        """
        if type(answer) == str:
            answer = sent_tokenize(answer)
        else:
            answer = [item.strip() for item in answer if item.strip() != ""]
        return answer

    def evaluation_generation(
        self,
        dataset: Dict,
        cand_key: str,
        ref_key: str,
        result_key: str,
    ) -> Dict:
        """
        Performs evalutions using rouge, bertScore and sentence bert

        Args:
            dataset (Dict): dataset with the candidate and reference to evaluate
            cand_key (str): key within the dict for the candidate
            ref_key (str): key within the dict for the reference
            result_key (str): where to store the result

        Returns:
            Dict: dataset with the stored result
        """
        # sentence transformer
        embedder = SentenceTransformer('all-mpnet-base-v2')
        # Bert scorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=True)

        # Initialize RougeScorer with split_summaries=True
        scorer_rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL', 'rougeLsum'],
            use_stemmer=True,
            split_summaries=True
        )
        handle_exceptions = self.collated_exceptions.new_handle_exception(
            result_key=result_key,
            action="evaluation",
            model_name="eval"
        )

        if cand_key not in dataset:
            print(f"{cand_key} not in dataset")
            print(dataset)
            exit()
        elif ref_key not in dataset:
            print(f"{ref_key} not in dataset")
            print(dataset)
            exit()

        dataset[cand_key] = ensure_string(dataset[cand_key], " ")
        dataset[ref_key] = ensure_string(dataset[ref_key], " ")

        if dataset[cand_key] == "" or dataset[ref_key] == "":
            dataset = self.log_eval_score(
                f"{result_key}_bertScore", dataset, [], is_blank=True)
            dataset = self.log_eval_score(
                f"{result_key}_sentence_transformer", dataset, [], is_blank=True)
            return dataset

        try:
            dataset[ref_key] = self.process_answer(dataset[ref_key])
            dataset[cand_key] = self.process_answer(dataset[cand_key])

            ref_window: List[str] = dataset[ref_key]
            cand_window: List[str] = dataset[cand_key]

            # bert-score
            bs_eval_list = self.eval_bert_score(
                scorer, cand_window, ref_window)

            # sentence transformers
            st_eval_list = self.eval_sentence_transformer(
                embedder=embedder,
                cand_window=cand_window,
                ref_window=ref_window
            )

            # Logging eval data to cand data
            dataset = self.log_eval_score(
                f"{result_key}_bertScore", dataset, bs_eval_list, is_blank=False)
            dataset = self.log_eval_score(
                f"{result_key}_sentence_transformer", dataset, st_eval_list, is_blank=False)

            # rouge eval
            scores_rouge = scorer_rouge.score(ensure_string(
                dataset[ref_key]), ensure_string(dataset[cand_key]))
            dataset[f"{result_key}_rouge1"] = scores_rouge['rouge1'].fmeasure
            dataset[f"{result_key}_rougeL"] = scores_rouge['rougeL'].fmeasure
            dataset[f"{result_key}_rougeLsum"] = scores_rouge['rougeLsum'].fmeasure

        except Exception as e:
            exception_content = {
                "reference": dataset[ref_key],
                "candidate": dataset[cand_key]
            }
            handle_exceptions.store_exceptions(exception_content, str(e))

        finally:
            return dataset

    @staticmethod
    def get_files_with_keyword(directory: str, keyword: str) -> List:
        """
        Get files from a directory with the matching keyword

        Args:
            directory (str): path of the directory
            keyword (str): keyword to look for

        Returns:
            List: returns a list of file paths
        """
        pattern = directory + '/**/*'
        file_list = glob.glob(pattern, recursive=True)

        # Filter the file list based on filename
        qa_files = [
            file for file in file_list if keyword in os.path.basename(file)]

        return qa_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory where data dir is stored",
    )
    parser.add_argument(
        "--data_file_path",
        type=str,
        required=True,
        help="The directory of the data dir",
    )
    return parser.parse_args()


if __name__ == "__main__":

    pass
