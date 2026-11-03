"""Stage 4 unit test: the two text normalisers on 10 fixed strings (preprocessing_validation.md)."""
import pytest

from src.asr.quality import wer_normalize
from src.common import normalize_text

# (input, normalize_text output, wer_normalize output)
CASES = [
    ("I’m fine.", "I'm fine.", "im fine"),                                  # curly apostrophe
    ("“Hello,” she said.", '"Hello," she said.', "hello she said"),   # curly double quotes
    ("Oh,   my   God!", "Oh, my God!", "oh my god"),                            # runs of spaces
    ("Well—maybe.", "Well-maybe.", "well maybe"),                          # em dash
    ("pre–war", "pre-war", "pre war"),                                     # en dash
    ("  leading and trailing  ", "leading and trailing", "leading and trailing"),
    ("Tab\tand\nnewline", "Tab and newline", "tab and newline"),
    ("What?!", "What?!", "what"),                                               # punctuation only stripped for WER
    ("Rachel Green", "Rachel Green", "rachel green"),                      # no-break space
    ("", "", ""),
]


@pytest.mark.parametrize("raw,keep,wer", CASES)
def test_normalizers(raw, keep, wer):
    assert normalize_text(raw) == keep
    assert wer_normalize(raw) == wer


def test_case_and_punctuation_kept_by_normalize_text():
    assert normalize_text("Hello, World!") == "Hello, World!"


def test_missing_values():
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""
    assert wer_normalize(None) == ""


def test_same_function_for_gold_and_asr():
    assert wer_normalize("It’s ME!") == wer_normalize("it's me")
