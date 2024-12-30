from typing import TypedDict, NotRequired


class MessagePacket(TypedDict):
    time: NotRequired[int]
    role: str
    content: str
