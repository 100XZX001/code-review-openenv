---
title: Code Review Environment
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.10"
app_file: app.py
pinned: false
---

# Code Review Environment

An OpenEnv environment for code review tasks. An AI agent must read a pull request (PR) code snippet, consider existing comments, and write a helpful comment that improves the code.

## Real-world relevance
Code review is a critical part of software development. Automating helpful code review can save developer time and improve code quality. This environment provides a simplified but realistic simulation for training agents to give constructive feedback.

## Action Space
- `write_comment` (with `comment_text`): Write a comment. The grader will evaluate its helpfulness.
- `skip`: End the episode without writing a comment (small penalty).
- `done`: End the episode early (larger penalty).

## Observation Space
- `pr_code` (string): The code snippet being reviewed.
- `comments` (list of strings): Existing comments on the PR.
- `agent_comment` (string, optional): The comment written by the agent (if any).
- `step` (int): Number of steps taken so far.
- `done` (bool): Whether the episode is finished.

## Tasks
| Task   | Description | Expected Keywords (partial credit) |
|--------|-------------|-----------------------------------|
| Easy   | Missing null check in dictionary lookup | null, key, missing, check, exists, handle |
| Medium | Inefficient loop; use `enumerate` | enumerate, for item in, range, inefficient, optimize |
| Hard   | Division by zero risk when list empty | empty, zero, length, check, handle, exception |

## Reward Function
- Writing a comment: +0.2 (dense) + quality score (0.0–1.0) from grader.
- Skipping: -0.1
- Ending early: -0.5
- Invalid action: -0.2

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the FastAPI server locally: `uvicorn server.app:app --reload`
3. Set environment variables (if testing inference): `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.

## Docker
Build the image:
```bash
docker build -t code-review-env .