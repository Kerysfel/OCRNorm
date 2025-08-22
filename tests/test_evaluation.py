import pytest

from src.evaluation import (
    preprocess_text,
    tokenize_words,
    tokenize_chars,
    levenshtein_distance,
)


def test_preprocess_text_basic():
    assert preprocess_text("Hello, World!") == "hello world"
    assert preprocess_text("  A\t B\nC  ") == "a b c"


def test_preprocess_text_only_punct():
    assert preprocess_text("!!!???...,,,") == ""


def test_preprocess_text_unicode():
    # Unicode letters should be preserved and lowercased; punctuation removed
    assert preprocess_text("Café — crème brûlée!") == "café crème brûlée"
    # Mixed quotes and dashes
    assert preprocess_text("“Quoted” — ‘text’") == "quoted text"


def test_tokenize_words_basic():
    assert tokenize_words("Hello, World!") == ["hello", "world"]
    assert tokenize_words("") == []


def test_tokenize_words_unicode_and_spaces():
    assert tokenize_words("  Café\t crème  ") == ["café", "crème"]


def test_tokenize_chars_basic():
    assert tokenize_chars("ABC") == ["a", "b", "c"]
    assert tokenize_chars("") == []


def test_tokenize_chars_unicode():
    assert tokenize_chars("Café") == list("café")


def test_levenshtein_distance_empty():
    assert levenshtein_distance([], []) == 0
    assert levenshtein_distance([], ["a"]) == 1
    assert levenshtein_distance(["a", "b"], []) == 2


def test_levenshtein_distance_words():
    a = ["the", "quick", "brown", "fox"]
    b = ["the", "fast", "brown", "fox"]
    assert levenshtein_distance(a, b) == 1


def test_levenshtein_distance_chars():
    a = list("kitten")
    b = list("sitting")
    assert levenshtein_distance(a, b) == 3


def test_levenshtein_distance_unicode():
    a = list("café")
    b = list("cafe")
    # One substitution (é -> e)
    assert levenshtein_distance(a, b) == 1
