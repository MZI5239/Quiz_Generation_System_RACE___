import os
import sys
import pytest

# Add src to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from inference import _generate_question_template
from preprocessing import tokenize

def test_tokenize():
    # Test basic tokenization
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenize(text, remove_stops=True)
    assert "quick" in tokens
    assert "fox" in tokens
    assert "the" not in tokens  # stopword should be removed

def test_generate_question_template():
    # Test that template generates appropriate question based on pronoun
    ans = "He went to the store"
    q = _generate_question_template("Dummy article about a person.", ans)
    assert "Who" in q
    
    ans_loc = "In the middle of the forest"
    q_loc = _generate_question_template("Dummy article about a forest.", ans_loc)
    assert "Where" in q_loc
