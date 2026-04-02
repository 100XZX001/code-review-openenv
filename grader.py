def grade_comment(comment: str, expected_keywords: list, task: str) -> float:
    """
    Returns a score in [0,1] based on keyword coverage and task-specific heuristics.
    """
    comment_lower = comment.lower()
    matched = sum(1 for kw in expected_keywords if kw in comment_lower)
    kw_score = min(1.0, matched / max(1, len(expected_keywords) // 2))

    # Bonus for length (≥ 15 words)
    words = comment.split()
    length_bonus = 0.1 if len(words) >= 15 else 0.0

    # Penalty for very short comments
    if len(words) < 5:
        penalty = 0.2
    else:
        penalty = 0.0

    # For hard tasks, also penalise if the comment is too vague
    if task in ["harder", "hardest"] and "lock" not in comment_lower and "thread" not in comment_lower:
        penalty += 0.1

    final = kw_score + length_bonus - penalty
    return max(0.0, min(1.0, final))

def grade_question(question: str) -> float:
    """
    Simple heuristic: longer, more specific questions get higher score.
    """
    words = question.split()
    if len(words) < 3:
        return 0.0
    # Check for question words
    if any(q in question.lower() for q in ["what", "how", "why", "where", "when", "does", "is"]):
        return min(1.0, len(words) / 20)  # up to 1.0
    return 0.2

def grade_fix(proposed_fix: str, expected_fix_keywords: list, hidden_test: callable) -> float:
    """
    Runs a simple test (if provided) and also checks keywords.
    For demonstration, we'll use a keyword‑based check, but in a real
    environment you'd execute tests.
    """
    # Keyword check
    matched = sum(1 for kw in expected_fix_keywords if kw in proposed_fix.lower())
    kw_score = min(1.0, matched / max(1, len(expected_fix_keywords) // 2))

    # If we have a real test function, run it
    test_score = 0.0
    if hidden_test is not None:
        try:
            test_score = hidden_test(proposed_fix)  # should return 0.0–1.0
        except Exception:
            test_score = 0.0

    # Weighted average: 60% tests, 40% keywords
    return 0.6 * test_score + 0.4 * kw_score
