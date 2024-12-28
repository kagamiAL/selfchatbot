import torch.nn.functional as F
import re


def filter_responses(prompt: str, responses: list[str]) -> list[str]:
    """Filters the responses so that only your response is returned

    Args:
        prompt (str): The prompt
        responses (list[str]): The responses

    Returns:
        list[str]: The filtered responses
    """
    pattern = rf"U:\s?{re.escape(prompt)}Y:\s?(.*?)(?=U:|$)"
    filtered = []
    for response in responses:
        result = re.findall(pattern, response.lstrip())
        filtered.append(result[-1])
    return filtered


class ChatModel:
    MAX_LENGTH = 1024
    history: list[str]
    generation_params: dict

    def __init__(self, model, tokenizer, params: dict):
        model.eval()
        self.history = []
        self.generation_params = params
        self.model = model
        self.tokenizer = tokenizer

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

    def __get_encoding(self, prompt: str, history: bool = False) -> dict[str, any]:
        """Returns the formatted encodings for the model

        Args:
            prompt (str): The prompt
            history (bool, optional): Whether or not to use chat history. Defaults to False.

        Returns:
            dict[str, any]: The formatted encodings
        """
        formatted_prompt = (
            f"{self.tokenizer.eos_token}U: {prompt}{self.tokenizer.eos_token}Y:"
        )
        final_prompt = formatted_prompt
        if history:
            final_prompt = "".join(self.history) + formatted_prompt
            i = 0
            max_attempts = len(self.history)
            while (
                len(self.tokenizer.encode(final_prompt)) > self.MAX_LENGTH
                and i < max_attempts
            ):
                i += 1
                final_prompt = "".join(self.history[i:]) + formatted_prompt
        return self.tokenizer.encode_plus(final_prompt, return_tensors="pt")

    def get_responses(self, input_ids) -> list[str]:
        """Returns the decoded responses from the model

        Args:
            input_ids (torch.tensor): input ids

        Returns:
            list[str]: list of responses from the model
        """

        sequences = self.__generate(input_ids).sequences

        # Decode the sequences
        return [
            self.tokenizer.decode(sequence, skip_special_tokens=True)
            for sequence in sequences
        ]

    def prompt_best_response(self, prompt: str, history: bool = False) -> str:
        """Prompts the model for the best response (calculated by log-probability)

        Args:
            prompt (str): The prompt
            history (bool, optional): Whether or not to use chat history. Defaults to False.

        Returns:
            str: The best response
        """
        prompt = prompt.strip()
        output = self.__generate(self.__get_encoding(prompt, history))
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

        # Decode the best sequence
        best_output = self.tokenizer.decode(best_sequence, skip_special_tokens=True)
        filtered = filter_responses(prompt, [best_output])
        if filtered:
            return filtered[0]
        return ""

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
        return filter_responses(
            passed_prompt,
            self.get_responses(self.__get_encoding(passed_prompt, history)),
        )

    def add_to_history(self, prompt: str, response: str):
        """Adds the prompt and response to the history

        Args:
            prompt (str): the prompt
            response (str): the response
        """
        self.history.append(
            f"{self.tokenizer.eos_token}U: {prompt.strip()}{self.tokenizer.eos_token}Y: {response}"
        )
