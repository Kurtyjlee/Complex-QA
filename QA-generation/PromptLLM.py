import openai
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class PromptLLM():
    """
    Object to execute LLM calls. Currently uses the FastChat API to prompt
    """
    def __init__(
        self,
        model_name: str,
        qa_config: Dict[Any, Any]
    ) -> None:
        """
        Constructor for PromptLLM

        Args:
            model_name (str): path to hugging face model or openai model
            qa_config (Dict[Any, Any]): qa_config file loading from the generation call
        """
        self.qa_config = qa_config

        self.chatcompletion_model = model_name
        openai.api_base = qa_config[model_name]["openai_localhost"]
        openai.api_key = qa_config[model_name]["openai_api_key"]
        openai.organization = qa_config[model_name]["openai_organization"]
        self.openai_completion = openai.ChatCompletion()

        # self.tokenizer, self.model = self.load_model("lmsys/vicuna-13b-v1.3") # Hugging face interface (deprecated)

    def check_if_model_exists(self, model_name: str) -> str:
        """
        Helps to check if the model exists in the config file

        Args:
            model_name (str): name of model to check

        Returns:
            str: if present, return the model name
        """
        if model_name in self.qa_config:
            print(f"{model_name} does not exist in config file")
            exit()
        return model_name

    def get_chat_model(self) -> str:
        """
        Returns the name of the model
        """
        return self.chatcompletion_model

    def current_model_name(self) -> str:
        """
        returns currently used model
        """
        return self.chatcompletion_model

    def prompt_model(
        self,
        definition: str,
        input: str,
        temp: int,
        max_tokens: int
    ) -> str:
        """
        Instructs the model with defintions and instructions

        Args:
            definition (str): Defines that the model should output
            input (str): Input for the model to do something
            temp (int): temp for the model
            max_tokens (int): max output tokens to generate

        Returns:
            str: returns the generation from the model
        """
        prompt = f"Input:{input}\nOutput:"

        output = self.openai_completion.create(
            model=self.chatcompletion_model,
            messages=[
                {"role": "system", "content": definition},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=max_tokens,
            do_sampling=True
        )

        output_text = output.choices[0].message.content

        return output_text
    
"""
Use of huggingface API to access model, uncomment to use the huggingface API, currently deprecated
"""

    # # Hugging face interface (deprecated)
    # @staticmethod
    # def load_model(
    #     model_name:str
    # ) -> tuple[Any, Any]:

    #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="left")
    #     model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    #     model.eval()
    #     eos_token_id = tokenizer.eos_token_id
    #     tokenizer.pad_token = tokenizer.eos_token
    #     model.config.pad_token_id = model.config.eos_token_id

    #     # Set pad_token_id to eos_token_id
    #     tokenizer.pad_token_id = eos_token_id
    #     tokenizer.padding_side = 'left'
    #     return tokenizer, model

    # Hugging face interface (deprecated)
    # @staticmethod
    # def hugging_prompt(
    #     input:str,
    #     max_tokens:int,
    #     tokenizer,
    #     model
    # ) -> str:
    #     input_train = tokenizer(input, return_tensors="pt")
    #     #print(input)
    #     input_train.to('cuda')
    #     #with torch.no_grad():
    #     #torch.cuda.empty_cache()
    #     with torch.no_grad():
    #         greedy_output = model.generate(
    #             input_ids=input_train['input_ids'],
    #             attention_mask=input_train['attention_mask'],
    #             num_beams=4,
    #             do_sample=True,
    #             top_p = 0.9,
    #             top_k = 40,
    #             max_new_tokens=max_tokens
    #         )
    #     decoded_output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
    #     return decoded_output
