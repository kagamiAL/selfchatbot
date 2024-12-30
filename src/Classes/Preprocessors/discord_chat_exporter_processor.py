import re
from Classes.Preprocessors.preprocessor import Preprocessor
from DataProcessing import functional_preprocessing as fp
from typing import override
from dateutil import parser
from Classes.TypeDicts import MessagePacket


def remove_meta_section(text: str) -> str:
    """Remove the meta section from a DiscordChatExporter export

    Args:
        text (str): the text to remove the meta section from

    Returns:
        str: the text with the meta section removed
    """
    pattern = r"=+\n.*?\n=+\n"
    return re.sub(pattern, "", text)


class DiscordChatExporterPreprocessor(Preprocessor):
    DATA_FORMAT: str = "DiscordChatExporter"
    username: str

    @override
    def __init__(self, params: dict):
        super().__init__(params)
        self.username = params["username"]

    @override
    def normalize(self, text: str) -> list[MessagePacket]:
        # We want to create a list of standard format before the final step of preprocessing
        # Standard format:
        # {
        #    "time": int,
        #    "role": str,
        #    "content": str
        # }
        pattern = r"\[\d{4}-\d{2}-\d{2} \d{1,2}:\d{2} [AP]M\]\s+(.*)"
        bracket_pattern = r"\[(.*?)\]"
        labels = self.formatter.get_labels()
        arr_text = [
            line.strip()
            for line in fp.default_cleanup(remove_meta_section(text)).splitlines()
        ]
        processed = []
        i = 0
        while i < len(arr_text):
            line = arr_text[i]
            match = re.search(pattern, line)
            if match:
                if (
                    i + 1 < len(arr_text)
                    and arr_text[i + 1]
                    and not re.search(pattern, arr_text[i + 1])
                ):
                    time_stamp = int(
                        parser.parse(
                            re.search(bracket_pattern, match.group(0)).group(1)
                        ).timestamp()
                    )
                    processed.append(
                        {
                            "time": time_stamp,
                            "role": labels[int(match.group(1) == self.username)],
                            "content": arr_text[i + 1],
                        }
                    )
                    i += 1
            elif processed:
                processed[-1]["content"] += f"\n{line}"
            i += 1
        return processed
