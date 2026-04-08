import os
import json
import textwrap
from typing import List

from openai import OpenAI
from environment import CodeReviewEnv
from models import Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "skip"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI code reviewer. Your task is to review a pull request.
    You can:
    - write a comment (prefixed with "write_comment:")
    - ask a clarifying question (prefixed with "ask_question:")
    - propose a fixed code snippet (prefixed with "propose_fix:")
    - skip if you cannot help (just "skip")
    - end the episode if the code is perfect (just "done")
    Be constructive, specific, and focus on improving code quality.
    Do not add any other text.
    """
).strip()


def build_user_prompt(step: int, obs: Observation, history: List[str]) -> str:
    newline = "\n"
    comments_str = newline.join(obs.comments) if obs.comments else "No existing comments"
    history_str = newline.join(history[-3:]) if history else "None"
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        PR Title: {obs.pr_title}
        Description: {obs.pr_description}
        Code to review:
        {obs.code_snippet}
        Conversation so far:
        {comments_str}
        Previous actions:
        {history_str}
        Your response (choose one of the following formats):
        - write_comment: [your comment]
        - ask_question: [your question]
        - propose_fix: [fixed code]
        - skip
        - done
        """
    ).strip()
    return prompt


def parse_model_action(response_text: str) -> Action:
    if not response_text:
        return Action(action_type=FALLBACK_ACTION)
    raw = response_text.strip()
    lower = raw.lower()
    if lower.startswith("skip"):
        return Action(action_type="skip")
    if lower.startswith("done"):
        return Action(action_type="done")
    if lower.startswith("write_comment"):
        if ":" in raw:
            comment = raw.split(":", 1)[1].strip()
        else:
            comment = raw[len("write_comment"):].strip()
        return Action(action_type="write_comment", comment_text=comment)
    if lower.startswith("ask_question"):
        if ":" in raw:
            question = raw.split(":", 1)[1].strip()
        else:
            question = raw[len("ask_question"):].strip()
        return Action(action_type="ask_question", question=question)
    if lower.startswith("propose_fix"):
        if ":" in raw:
            fix = raw.split(":", 1)[1].strip()
        else:
            fix = raw[len("propose_fix"):].strip()
        return Action(action_type="propose_fix", fix_code=fix)
    return Action(action_type="write_comment", comment_text=raw)


def main() -> None:
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        # This error message is printed to stderr? It must not appear on stdout.
        # Use sys.stderr to avoid polluting structured output.
        import sys
        print("Error: API_BASE_URL, HF_TOKEN/API_KEY, and MODEL_NAME must be set.", file=sys.stderr)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CodeReviewEnv()
    tasks = ["easy", "medium", "hard", "harder", "hardest"]

    for task in tasks:
        env.set_task(task)
        obs = env.reset()
        history: List[str] = []
        done = False
        step = 0
        final_reward = 0.0

        # START marker
        print(f"[START] task={task}", flush=True)

        while not done and step < MAX_STEPS:
            step += 1
            user_prompt = build_user_prompt(step, obs, history)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                # Print error to stderr only
                import sys
                print(f"Request failed: {exc}. Using fallback.", file=sys.stderr)
                response_text = FALLBACK_ACTION

            action = parse_model_action(response_text)
            obs, reward, done, info = env.step(action)
            final_reward = reward.value

            # STEP marker with required key=value format
            print(f"[STEP] step={step} reward={final_reward:.3f}", flush=True)

            history.append(f"Step {step}: {action.action_type}")

        # END marker
        print(f"[END] task={task} score={final_reward:.3f} steps={step}", flush=True)


if __name__ == "__main__":
    main()
