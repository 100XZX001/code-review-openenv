from pydantic import BaseModel
from typing import List, Optional

class Observation(BaseModel):
    pr_code: str
    comments: List[str]
    agent_comment: Optional[str] = None
    step: int = 0
    done: bool = False

class Action(BaseModel):
    action_type: str  # "write_comment", "skip", "done"
    comment_text: Optional[str] = None

class Reward(BaseModel):
    value: float

class State(BaseModel):
    pr_code: str
    comments: List[str]
    agent_comment: Optional[str]
    step: int
    done: bool