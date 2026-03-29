import os
import json
import textwrap
from typing import List

from openai import OpenAI
from environment import CodeReviewEnv
from models import Action, Observation

# Environment variables (required by hackathon)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = "skip"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI code reviewer. Your task is to provide helpful comments on pull requests.
    You will see a code snippet and existing comments.
    
    Reply with ONE of the following:
    - "write_comment: [your comment]" - to provide a helpful code review comment
    - "skip" - if you cannot provide a helpful comment
    - "done" - if the code is already perfect
    
    Be constructive, specific, and focus on improving code quality.
    """
).strip()

def build_user_prompt(step: int, obs: Observation, history: List[str]) -> str:
    newline = "\n"
    comments_str = newline.join(obs.comments) if obs.comments else "No existing comments"
    history_str = newline.join(history[-3:]) if history else "None"
    
    prompt = textwrap.dedent(
        f"""
        Step: {step}
        
        Code to review:
        {obs.pr_code}

        Existing comments on this PR:
        {comments_str}

        Previous actions:
        {history_str}

        Please provide your response (write_comment, skip, or done):
        """
    ).strip()
    return prompt

def parse_model_action(response_text: str) -> Action:
    if not response_text:
        return Action(action_type=FALLBACK_ACTION)

    raw_text = response_text.strip()
    lower_text = raw_text.lower()

    if lower_text.startswith("skip"):
        return Action(action_type="skip")
    if lower_text.startswith("done"):
        return Action(action_type="done")
    if lower_text.startswith("write_comment"):
        if ":" in raw_text:
            comment = raw_text.split(":", 1)[1].strip()
        else:
            comment = raw_text[len("write_comment"):].strip()
        if not comment:
            return Action(action_type="skip")
        return Action(action_type="write_comment", comment_text=comment)
    # default: treat as a comment
    return Action(action_type="write_comment", comment_text=raw_text)

def main() -> None:
    if not API_BASE_URL or not API_KEY or not MODEL_NAME:
        print("Error: API_BASE_URL, HF_TOKEN/API_KEY, and MODEL_NAME must be set.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = CodeReviewEnv()
    tasks = ["easy", "medium", "hard"]
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
            
            history.append(f"Step {step}: {action.action_type}")
            print(f"  Step {step} | Action: {action.action_type} | Reward: {reward.value:.2f}")

        scores[task] = final_reward
        print(f"{task.upper()} completed. Final Score: {final_reward:.2f}")

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print(json.dumps(scores, indent=2))
    print("="*50)

if __name__ == "__main__":
    main()