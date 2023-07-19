#!/bin/bash

CONDA_PYTHON_PATH="path/to/python"

echo running train script
$CONDA_PYTHON_PATH close-book-generation.py \
    --context_name nyt \
    --questions_path ../data/generations/nyt/questions_nyt_nyt_data_2021_vicuna-13b-v1.3.json \
    --num_of_generations 100 \
    --model_name vicuna-13b-v1.3 \
    --qa_config ./configs/QA_config.yaml \
    --identifier 2021_batch \
