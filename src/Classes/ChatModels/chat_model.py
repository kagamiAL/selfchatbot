import torch.nn.functional as F
from collections import deque
from Classes.Formatters.formatter import Formatter, get_formatter
from Classes.TypeDicts import MessagePacket
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatModel:
    MAX_LENGTH = 1024
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    history: list[MessagePacket]
    generation_params: dict
    formatter: Formatter

    def __init__(self, model, tokenizer, params: dict):
        model.eval()
        self.history = []
        self.generation_params = params
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = get_formatter(params["model"], self.tokenizer)
        self.MAX_LENGTH = params["max_length"]
        self.MAX_LENGTH = min(self.MAX_LENGTH, self.tokenizer.model_max_length)

    def __get_encoding(self, prompt: str, history: bool = False):
        """
        Get the encoding of the prompt, possibly with the history

        Args:
            prompt (str): The prompt to encode
            history (bool, optional): Whether to include the history in the encoding. Defaults to False.

        Raises:
            ValueError: If the prompt is too long

        Returns:
            dict: a dictionary with the encoding
        """
        block = deque([])
        if history:
            block += self.history
        block.append({"role": self.formatter.USER_LABEL, "content": prompt})
        encoding: dict = self.formatter.get_prompt_encoding(block)
        while len(encoding["input_ids"]) > self.MAX_LENGTH:
            block.popleft()
            encoding = self.formatter.get_prompt_encoding(block)
        if not block:
            raise ValueError("Prompt is too long for given max length")
        return encoding

    def __generate(self, input_ids):
        """Generates from the model

        Args:
            input_ids (torch.tensor): input ids

        Returns:
            output: output from model
        """
        return self.model.generate(
            **input_ids,
            max_length=self.MAX_LENGTH,  # Maximum response length
            temperature=self.generation_params["temperature"],
            top_p=self.generation_params["top_p"],
            top_k=self.generation_params["top_k"],
            do_sample=True,
            repetition_penalty=self.generation_params["repetition_penalty"],
            num_return_sequences=self.generation_params["num_return_sequences"],
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    def prompt_best_response(self, prompt: str, history: bool = False) -> str:
        """Prompts the model for the best response (calculated by log-probability)

        Args:
            prompt (str): The prompt
            history (bool, optional): Whether or not to use chat history. Defaults to False.

        Returns:
            str: The best response
        """
        prompt = prompt.strip()
        encoded_prompt = self.__get_encoding(prompt, history)
        output = self.__generate(encoded_prompt)
        # Decode sequences and extract token-wise scores
        sequences = output.sequences
        scores = output.scores  # List of token logits at each step

        # Calculate log-probabilities for each sequence
        log_probs = []
        for seq_idx, seq in enumerate(sequences):
            log_prob = 0
            for step_idx, token_logits in enumerate(scores):
                token_id = seq[step_idx]
                token_prob = F.log_softmax(token_logits, dim=-1)[seq_idx, token_id]
                log_prob += token_prob.item()
            log_probs.append(log_prob)

        # Normalize scores by sequence length
        normalized_scores = [
            score / len(seq) for score, seq in zip(log_probs, sequences)
        ]

        # Find the best sequence
        best_sequence_idx = normalized_scores.index(max(normalized_scores))
        best_sequence = sequences[best_sequence_idx]

        print(f"DEBUG: {repr(self.tokenizer.decode(best_sequence))}")

        return self.formatter.filter_output(
            encoded_prompt["input_ids"].squeeze(0), best_sequence
        )

    def prompt_for_responses(
        self, passed_prompt: str, history: bool = False
    ) -> list[str]:
        """Prompts the model for a response

        Args:
            prompt (str): The prompt to use
            history (bool, optional): Whether or not to use chat history. Defaults to False.

        Returns:
            list[str]: The responses from the model
        """
        passed_prompt = passed_prompt.strip()
        encoding = self.__get_encoding(passed_prompt, history)
        return [
            self.formatter.filter_output(encoding["input_ids"].squeeze(0), seq)
            for seq in self.__generate(encoding)
        ]

    def add_to_history(self, prompt: str, response: str):
        """Adds the prompt and response to the history

        Args:
            prompt (str): the prompt
            response (str): the response
        """
        self.history.extend(
            [
                {"role": self.formatter.USER_LABEL, "content": prompt},
                {"role": self.formatter.MODEL_LABEL, "content": response},
            ]
        )
