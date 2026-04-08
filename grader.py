import numpy as np
from sentence_transformers import SentenceTransformer, util

EPS = 0.001

def clamp_score(score):
    if score <= 0.0:
        return EPS
    if score >= 1.0:
        return 1.0 - EPS
    return score

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def grade_comment(comment: str, expected_keywords: list, expert_comment: str) -> float:
    if not comment:
        return clamp_score(0.0)
    model = _get_model()
    emb_comment = model.encode(comment, convert_to_tensor=True)
    emb_expert = model.encode(expert_comment, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_comment, emb_expert).item()
    comment_lower = comment.lower()
    matched = sum(1 for kw in expected_keywords if kw in comment_lower)
    kw_score = min(1.0, matched / max(1, len(expected_keywords) // 2))
    combined = 0.7 * sim + 0.3 * kw_score
    words = comment.split()
    if len(words) >= 15:
        length_bonus = 0.1
    elif len(words) < 5:
        length_bonus = -0.2
    else:
        length_bonus = 0.0
    final = combined + length_bonus
    # Clamp to (0,1) using EPS
    return clamp_score(final)

def grade_question(question: str) -> float:
    words = question.split()
    if len(words) < 3:
        return clamp_score(0.0)
    if any(q in question.lower() for q in ["what", "how", "why", "where", "when", "does", "is"]):
        score = min(1.0, len(words) / 20)
        return clamp_score(score)
    return clamp_score(0.2)

def grade_fix(proposed_fix: str, expected_fix_keywords: list, hidden_test: callable) -> float:
    matched = sum(1 for kw in expected_fix_keywords if kw in proposed_fix.lower())
    kw_score = min(1.0, matched / max(1, len(expected_fix_keywords) // 2))
    test_score = 0.0
    if hidden_test is not None:
        try:
            test_score = hidden_test(proposed_fix)
        except Exception:
            test_score = 0.0
    score = 0.6 * test_score + 0.4 * kw_score
    return clamp_score(score)
