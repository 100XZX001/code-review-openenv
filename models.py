from pydantic import BaseModel
from typing import List, Optional

class Action(BaseModel):
    action_type: str   # "write_comment", "skip", "done", "ask_question", "propose_fix"
    comment_text: Optional[str] = None
    question: Optional[str] = None          # for ask_question
    fix_code: Optional[str] = None          # for propose_fix

class Observation(BaseModel):
    pr_title: str
    pr_description: str
    code_snippet: str
    comments: List[str]                     # conversation history (includes agent questions and environment answers)
    test_results: Optional[str] = None      # output of running tests on proposed fix
    step: int = 0
    done: bool = False

class Reward(BaseModel):
    value: float

class State(BaseModel):
    pr_title: str
    pr_description: str
    code_snippet: str
    comments: List[str]
    test_results: Optional[str]
    step: int
    done: bool
