from environment import CodeReviewEnv
from models import Action, Observation

def run_task(task_name: str, agent_func) -> float:
    """
    Run a single task episode with the given agent function.
    
    Args:
        task_name: one of 'easy', 'medium', 'hard'
        agent_func: a callable that takes an Observation and returns an Action.
    
    Returns:
        float: the final reward (score) from the environment.
    """
    env = CodeReviewEnv()
    env.set_task(task_name)
    obs = env.reset()
    done = False
    step = 0
    final_reward = 0.0

    while not done and step < 5:   # max 5 steps
        step += 1
        action = agent_func(obs)
        obs, reward, done, info = env.step(action)
        final_reward = reward.value

    return final_reward