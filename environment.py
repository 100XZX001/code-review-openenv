from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, State

class CodeReviewEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self.reset()

    def set_task(self, task: str):
        if task not in ["easy", "medium", "hard"]:
            raise ValueError(f"Unknown task: {task}")
        self.task = task

    def reset(self) -> Observation:
        if self.task is None:
            raise RuntimeError("Task not set. Call set_task() first.")
        self.step_count = 0
        self.agent_comment = None
        self.done = False

        if self.task == "easy":
            self.pr_code = "def get_user(id):\n    return users[id]  # missing null check"
            self.comments = ["Looks good!", "Maybe add a comment?"]
        elif self.task == "medium":
            self.pr_code = "for i in range(len(items)):\n    process(items[i])\n# O(n^2) when it could be O(n)"
            self.comments = ["Nice code"]
        elif self.task == "hard":
            self.pr_code = "def calculate_average(data):\n    total = sum(data)\n    return total / len(data)  # what if data is empty?"
            self.comments = ["LGTM"]
        else:
            raise RuntimeError(f"Invalid task: {self.task}")

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already finished")

        reward = 0.0
        info = {}

        if action.action_type == "write_comment":
            self.agent_comment = action.comment_text or ""
            reward = 0.2
            quality_score = self._grade_comment(self.agent_comment)
            reward += quality_score
            if reward > 1.0:
                reward = 1.0
            self.done = True
        elif action.action_type == "skip":
            reward = -0.1
            self.done = True
        elif action.action_type == "done":
            reward = -0.5
            self.done = True
        else:
            reward = -0.2
            self.done = True

        self.step_count += 1
        obs = self._get_observation()
        return obs, Reward(value=reward), self.done, info

    def _grade_comment(self, comment: str) -> float:
        if self.task == "easy":
            keywords = ["null", "key", "missing", "check", "exists", "handle"]
            matched = sum(1 for kw in keywords if kw in comment.lower())
            return min(1.0, matched / 3)
        elif self.task == "medium":
            keywords = ["enumerate", "for item in", "range", "inefficient", "optimize"]
            matched = sum(1 for kw in keywords if kw in comment.lower())
            return min(1.0, matched / 3)
        elif self.task == "hard":
            keywords = ["empty", "zero", "length", "check", "handle", "exception"]
            matched = sum(1 for kw in keywords if kw in comment.lower())
            return min(1.0, matched / 3)
        else:
            return 0.0

    def _get_observation(self) -> Observation:
        return Observation(
            pr_code=self.pr_code,
            comments=self.comments,
            agent_comment=self.agent_comment,
            step=self.step_count,
            done=self.done
        )

    def state(self) -> State:
        return State(
            pr_code=self.pr_code,
            comments=self.comments,
            agent_comment=self.agent_comment,
            step=self.step_count,
            done=self.done
        )