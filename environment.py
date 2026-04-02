from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, State
from grader import grade_comment, grade_question, grade_fix

class CodeReviewEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self.reset()

    def set_task(self, task: str):
        if task not in ["easy", "medium", "hard", "harder", "hardest"]:
            raise ValueError(f"Unknown task: {task}")
        self.task = task

    def reset(self) -> Observation:
        if self.task is None:
            raise RuntimeError("Task not set. Call set_task() first.")
        self.step_count = 0
        self.agent_comment = None
        self.done = False
        self.test_results = None

        # Task definitions (same as before)
        if self.task == "easy":
            self.pr_title = "Fix missing null check in user lookup"
            self.pr_description = "The current code does not handle missing user IDs. It raises a KeyError."
            self.code_snippet = "def get_user(id):\n    return users[id]  # missing null check"
            self.comments = []
            self.expected_keywords = ["null", "key", "missing", "check", "exists", "handle"]
            self.expert_comment = "Add a check to ensure the key exists before accessing the dictionary to avoid KeyError."
            self.expected_fix_keywords = ["if id in users"]
        elif self.task == "medium":
            self.pr_title = "Improve loop efficiency"
            self.pr_description = "The loop uses `range(len(items))` which is inefficient and less readable."
            self.code_snippet = "for i in range(len(items)):\n    process(items[i])\n# O(n^2) when it could be O(n)"
            self.comments = []
            self.expected_keywords = ["enumerate", "for item in", "range", "inefficient", "optimize"]
            self.expert_comment = "Use `for item in items:` for a more Pythonic and efficient loop."
            self.expected_fix_keywords = ["for item in items", "for i, item in enumerate"]
        elif self.task == "hard":
            self.pr_title = "Handle division by zero in average calculation"
            self.pr_description = "The function crashes when the input list is empty."
            self.code_snippet = "def calculate_average(data):\n    total = sum(data)\n    return total / len(data)  # what if data is empty?"
            self.comments = []
            self.expected_keywords = ["empty", "zero", "length", "check", "handle", "exception"]
            self.expert_comment = "Check if the list is empty and return a sensible default (e.g., 0) or raise a descriptive error."
            self.expected_fix_keywords = ["if not data", "if len(data)==0"]
        elif self.task == "harder":
            self.pr_title = "Fix race condition in counter increment"
            self.pr_description = "Multiple threads increment a counter without synchronization, causing lost updates."
            self.code_snippet = "counter = 0\ndef increment():\n    global counter\n    counter += 1\n# called from multiple threads"
            self.comments = []
            self.expected_keywords = ["thread", "lock", "synchronization", "atomic", "race", "concurrent"]
            self.expert_comment = "Use a threading.Lock to protect the counter increment, or use an atomic operation like `threading.atomic`."
            self.expected_fix_keywords = ["lock", "threading.Lock", "with lock"]
        else:  # hardest
            self.pr_title = "Fix deadlock in database transaction"
            self.pr_description = "Two threads acquire locks in opposite order, leading to potential deadlock."
            self.code_snippet = "with lock1:\n    with lock2:\n        do_work()\n# another thread does lock2 then lock1"
            self.comments = []
            self.expected_keywords = ["deadlock", "lock order", "acquire", "release", "trylock", "timeout"]
            self.expert_comment = "Ensure all threads acquire locks in the same order to prevent deadlock. Consider using a timeout or a single lock."
            self.expected_fix_keywords = ["same order", "lock order", "acquire lock1 then lock2"]

        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode already finished")

        reward = 0.0
        info = {}

        if action.action_type == "write_comment":
            self.agent_comment = action.comment_text or ""
            reward = 0.2  # dense bonus for writing
            quality_score = grade_comment(self.agent_comment, self.expected_keywords, self.task)
            reward += quality_score
            self.done = True

        elif action.action_type == "ask_question":
            if not action.question:
                reward = -0.1
            else:
                q_score = grade_question(action.question)
                reward = 0.1 + q_score   # small bonus + quality
                # Simulate a helpful answer
                answer = self._answer_question(action.question)
                self.comments.append(f"Agent: {action.question}")
                self.comments.append(f"Env: {answer}")
                self.step_count += 1
                # Episode continues, not done

        elif action.action_type == "propose_fix":
            if not action.fix_code:
                reward = -0.2
            else:
                # We'll use a simple keyword check for demonstration
                # In a full version, you'd run unit tests
                fix_score = grade_fix(action.fix_code, self.expected_fix_keywords, None)
                reward = 0.3 + fix_score
                self.test_results = f"Fix evaluated with score {fix_score:.2f}"
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

    def _answer_question(self, question: str) -> str:
        # Simple rule‑based answers – you can expand
        q = question.lower()
        if "what" in q and "purpose" in q:
            return "The purpose of this function is to retrieve a user by ID from a dictionary."
        elif "expected" in q:
            return "The function should return the user object if the ID exists, otherwise raise a KeyError."
        elif "how" in q and "fix" in q:
            return "You might consider adding a check for missing keys or using a safer dictionary method like `get`."
        else:
            return "I'm not sure. Could you be more specific?"

    def _get_observation(self) -> Observation:
        return Observation(
            pr_title=self.pr_title,
            pr_description=self.pr_description,
            code_snippet=self.code_snippet,
            comments=self.comments.copy(),
            test_results=self.test_results,
            step=self.step_count,
            done=self.done
        )

    def state(self) -> State:
        return State(
            pr_title=self.pr_title,
            pr_description=self.pr_description,
            code_snippet=self.code_snippet,
            comments=self.comments.copy(),
            test_results=self.test_results,
            step=self.step_count,
            done=self.done
        )
