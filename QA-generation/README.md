# QA-generation
This directory contains the script for question and answer generations.

<pre>
📦QA-generation
 ┣ 📜HandleExceptions.py
 ┣ 📜PromptLLM.py
 ┣ 📜QaController.py
 ┣ 📜QaGeneration.py
 ┣ 📜close-book-generation.py
 ┣ 📜evaluation.py
 ┣ 📜open-book-generation.py
 ┣ 📜perplexity.py
 ┗ 📜question-generation.py
</pre>

## Graph representation on program flow: 
#### Performing question generation:
question-generation.py ➜ QaController.py ➜ QaGeneration ➜ PromptLLM.py & HandleExceptions.py

#### Performing close-book answer generation:
close-book-generation.py ➜ QaController.py ➜ QAGeneration ➜ PromptLLM.py & HandleExceptions.py ➜ evaluation.py

#### Calculating perplexity:
perplexity.py