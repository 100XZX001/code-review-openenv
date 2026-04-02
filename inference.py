import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from environment import CodeReviewEnv
from models import Action, Observation

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "skip"   # default action if model fails

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
    """Build the user prompt from the observation."""
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
    """Parse the model's response into an Action object."""
    if not response_text:
        return Action(action_type=FALLBACK_ACTION)

    raw = response_text.strip()
    lower = raw.lower()

    if lower.startswith("skip"):
        return Action(action_type="skip")
    if lower.startswith("done"):
        return Action(action_type="done")

    if lower.startswith("write_comment"):
        # Extract comment after the colon
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

    # If none matched, treat the whole response as a comment (fallback)
    return Action(action_type="write_comment", comment_text=raw)


def main() -> None:
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        print("Error: API_BASE_URL, HF_TOKEN/API_KEY, and MODEL_NAME must be set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CodeReviewEnv()
    tasks = ["easy", "medium", "hard", "harder", "hardest"]
    scores = {}

    print("=" * 50)
    print("Code Review Environment - Baseline Inference")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("=" * 50)

    for task in tasks:
        print(f"\nRunning task: {task.upper()}")
        env.set_task(task)
        obs = env.reset()
        history: List[str] = []
        done = False
        step = 0
        final_reward = 0.0

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
                print(f"  Request failed: {exc}. Using fallback.")
                response_text = FALLBACK_ACTION

            action = parse_model_action(response_text)
            obs, reward, done, info = env.step(action)
            final_reward = reward.value

            # Log the step
            log = f"Step {step}: {action.action_type}"
            if action.comment_text:
                log += f" comment: {action.comment_text[:50]}"
            if action.question:
                log += f" question: {action.question[:50]}"
            if action.fix_code:
                log += f" fix: {action.fix_code[:50]}"
            history.append(log)
            print(f"  {log} | Reward: {reward.value:.2f}")

        scores[task] = final_reward
        print(f"{task.upper()} completed. Final Score: {final_reward:.2f}")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print(json.dumps(scores, indent=2))
    print("="*50)


if __name__ == "__main__":
    main()
