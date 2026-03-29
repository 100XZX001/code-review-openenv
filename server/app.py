import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from typing import Optional
from environment import CodeReviewEnv
from models import Action, Observation, State

app = FastAPI(title="Code Review Environment", version="1.0.0")
env = CodeReviewEnv()  # default task "easy" is set in __init__

# ------------------------- Simple root endpoint -------------------------
@app.get("/")
def root():
    return {"message": "......Code Review Environment is running....by yuvraj"}

# ------------------------- OpenEnv required endpoints -------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "Code Review Environment",
        "description": "Simulate reviewing a PR and adding helpful comments. Three tasks with increasing difficulty."
    }

@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema()
    }

@app.post("/mcp")
def mcp():
    # Minimal JSON-RPC 2.0 response
    return {"jsonrpc": "2.0", "result": None}

# ------------------------- Your environment endpoints -------------------------
@app.post("/reset")
def reset(task: Optional[str] = None):
    if task:
        try:
            env.set_task(task)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    obs = env.reset()
    return obs.dict()

@app.post("/set_task")
def set_task(task: str):
    try:
        env.set_task(task)
        return {"status": "ok", "task": task}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state().dict()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()