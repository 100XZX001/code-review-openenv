import os
import sys
import textwrap

# ----------------------------------------------------------------------
#  Always define fallback values – do NOT exit early
# ----------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://dummy.api")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
MAX_STEPS = 5
FALLBACK_ACTION = "skip"

# We'll import the environment only after setting dummy env vars
from environment import CodeReviewEnv
from models import Action

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI code reviewer. Reply with one of:
    - write_comment: [comment]
    - ask_question: [question]
    - propose_fix: [code]
    - skip
    - done
    """
).strip()

def build_user_prompt(step, obs, history):
    return f"Step {step}\nCode:\n{obs.code_snippet}\nComments:\n{obs.comments}\nHistory:\n{history}\nYour response:"

def parse_model_action(text):
    if not text:
        return Action(action_type=FALLBACK_ACTION)
    lower = text.strip().lower()
    if lower.startswith("skip"):
        return Action(action_type="skip")
    if lower.startswith("done"):
        return Action(action_type="done")
    if lower.startswith("write_comment"):
        comment = text.split(":", 1)[1].strip() if ":" in text else text[14:].strip()
        return Action(action_type="write_comment", comment_text=comment)
    if lower.startswith("ask_question"):
        question = text.split(":", 1)[1].strip() if ":" in text else text[12:].strip()
        return Action(action_type="ask_question", question=question)
    if lower.startswith("propose_fix"):
        fix = text.split(":", 1)[1].strip() if ":" in text else text[11:].strip()
        return Action(action_type="propose_fix", fix_code=fix)
    return Action(action_type="write_comment", comment_text=text)

def main():
    # Even if API credentials are missing, we still run the loop with fallback actions.
    # We'll try to create an OpenAI client only if the base URL seems valid.
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_BASE_URL != "https://dummy.api" else None
    except Exception:
        client = None

    env = CodeReviewEnv()
    tasks = ["easy", "medium", "hard", "harder", "hardest"]

    for task in tasks:
        env.set_task(task)
        obs = env.reset()
        history = []
        done = False
        step = 0
        final_reward = 0.0

        # [START] marker
        sys.stdout.write(f"[START] task={task}\n")
        sys.stdout.flush()

        while not done and step < MAX_STEPS:
            step += 1
            prompt = build_user_prompt(step, obs, history)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

            # Try to get a model response, but fallback immediately if client is None or fails
            response_text = FALLBACK_ACTION
            if client is not None:
                try:
                    resp = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=200,
                    )
                    response_text = resp.choices[0].message.content or FALLBACK_ACTION
                except Exception:
                    # Any API error → fallback
                    pass

            action = parse_model_action(response_text)
            obs, reward, done, _ = env.step(action)
            final_reward = reward.value

            # [STEP] marker
            sys.stdout.write(f"[STEP] step={step} reward={final_reward:.3f}\n")
            sys.stdout.flush()

            history.append(f"Step {step}: {action.action_type}")

        # [END] marker
        sys.stdout.write(f"[END] task={task} score={final_reward:.3f} steps={step}\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
