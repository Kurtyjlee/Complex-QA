import argparse
import datetime
import sys
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from typing import List, Dict
import yaml
from QaGeneration import ensure_string
import statistics
from nltk.tokenize import sent_tokenize

"""
Usage: python3 perplexity.py --perplexity_config ../configs/perplexity_filepath.yml
"""

class Perplexity():
    """
    Object to calculate Perplexity
    """
    def __init__(self, perplexity_config: str):
        """
        Constructor for Perplexity calculations

        Args:
            perplexity_config (str): file path to the perplexity config
        """
        try:
            with open(perplexity_config, "r") as f:
                self.per_config = yaml.safe_load(f)
        except Exception as e:
            print(str(e))
            print("--perplexity_config has to be a yaml file")
            sys.exit()

        self.device_map = None
        with open(self.per_config["device_map"], "r") as f:
            self.device_map = json.load(f)

        if self.device_map == None:
            print("error loading device map, please check the yaml config file")

        self.device = self.per_config["model_name"]

        self.file_paths = self.per_config["filepaths"]

        memory_map = {
            0: "16GB",
            1: "15GB",
            2: "15GB",
            3: "17GB"
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.device,
            device_map="auto",
            max_memory=memory_map
        )

        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.device, use_fast=True)

    def execute_perplexity_calc(self) -> None:
        """
        Calculates perplexity based on the file paths in the perplexity config
        """
        for file in self.file_paths:
            with open(file, "r") as f:
                dataset: List[Dict[str, str]] = json.load(f)

            perplexity_file = f"{self.per_config['store_dir']}/perplexity_{file.split('/')[-1]}"

            perplexity_total = []

            master_context_set = set()
            for data in dataset:
                master_context_set.add(ensure_string(
                    data["context"], joiner=". "))
            master_context_list = list(master_context_set)

            progress_bar = tqdm.tqdm(
                total=len(master_context_list),
                desc=file.split("/")[-1]
            )

            for context in master_context_list:
                if context == "":
                    continue
                try:
                    context_list = sent_tokenize(context)

                    # Calculating perplexity spread
                    perplexity_spread = []
                    for index, text in enumerate(context_list):
                        with torch.no_grad():
                            progress_bar.set_postfix(
                                {'Info': f"calculating list {index}"})
                            inputs = self.tokenizer(text, return_tensors="pt")
                            loss = self.model(
                                input_ids=inputs["input_ids"], labels=inputs["input_ids"]
                            ).loss
                            ppl = torch.exp(loss).item()
                            perplexity_spread.append(ppl)

                    # Calculating perplexity overall
                    with torch.no_grad():
                        progress_bar.set_postfix(
                            {'Info': f"Calculating overall"})
                        inputs = self.tokenizer(context, return_tensors="pt")
                        loss = self.model(
                            input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
                        perplexity_overall = torch.exp(loss).item()

                    perplexity_total.append({
                        "context": context,
                        "perplexity_spread": perplexity_spread,
                        "perplexity_mean": statistics.mean(perplexity_spread),
                        "perplexity_overall": perplexity_overall
                    })
                    progress_bar.update(1)
                    torch.cuda.empty_cache()

                    with open(perplexity_file, "w") as f:
                        json.dump(perplexity_total, f, indent=2)
                except Exception as e:
                    with open(self.per_config["store_dir"], "a") as f:
                        f.writelines(f"{datetime.datetime.now()}: {str(e)}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perplexity_config",
        type=str,
        required=True,
        help="path to the perplexity config file",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    perplexity = Perplexity(args.perplexity_config)
    perplexity.execute_perplexity_calc()
