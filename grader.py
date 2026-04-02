import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load model once globally
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def grade_comment(comment: str, expected_keywords: list, expert_comment: str) -> float:
    """
    Returns a score in [0,1] based on:
    - semantic similarity with expert comment (70%)
    - keyword coverage (30%)
    - length bonus/penalty
    """
    if not comment:
        return 0.0

    # 1. Semantic similarity
    model = _get_model()
    emb_comment = model.encode(comment, convert_to_tensor=True)
    emb_expert = model.encode(expert_comment, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_comment, emb_expert).item()  # in [0,1]

    # 2. Keyword coverage
    comment_lower = comment.lower()
    matched = sum(1 for kw in expected_keywords if kw in comment_lower)
    kw_score = min(1.0, matched / max(1, len(expected_keywords) // 2))

    # 3. Combine (70% semantic, 30% keywords)
    combined = 0.7 * sim + 0.3 * kw_score

    # 4. Length bonus/penalty
    words = comment.split()
    if len(words) >= 15:
        length_bonus = 0.1
    elif len(words) < 5:
        length_bonus = -0.2
    else:
        length_bonus = 0.0

    # 5. Final score, clamped
    final = combined + length_bonus
    return max(0.0, min(1.0, final))


def grade_question(question: str) -> float:
    """Simple heuristic for question quality."""
    words = question.split()
    if len(words) < 3:
        return 0.0
    # Check for question words
    if any(q in question.lower() for q in ["what", "how", "why", "where", "when", "does", "is"]):
        return min(1.0, len(words) / 20)
    return 0.2


def grade_fix(proposed_fix: str, expected_fix_keywords: list, hidden_test: callable) -> float:
    """Evaluates a code fix. Hidden_test can be a function that runs unit tests."""
    # Keyword check (simplified)
    matched = sum(1 for kw in expected_fix_keywords if kw in proposed_fix.lower())
    kw_score = min(1.0, matched / max(1, len(expected_fix_keywords) // 2))

    # If we have a test function, run it
    test_score = 0.0
    if hidden_test is not None:
        try:
            test_score = hidden_test(proposed_fix)
        except Exception:
            test_score = 0.0

    # Weighted: 60% test, 40% keywords
    return 0.6 * test_score + 0.4 * kw_score
