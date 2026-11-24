"""Force every leakage detector to fire on a planted case, and prove clean synthetic intents pass."""
import pytest

from wa.leakage import check, load_test

TEST = load_test()
T0 = TEST[0]  # "Get the top-1 best-selling product name(s) in 2022", template "Get the top-{{n}} best-selling {{entity}} in {{period}}"


def kinds(train, embed=False):
    return [h["kind"] for h in check(train, TEST, embed=embed)]


def test_exact_fires_despite_case_and_punctuation():
    assert kinds([{"intent": "  get the TOP-1 best-selling product name(s) in 2022!! "}]) == ["exact"]


def test_template_fires_on_new_instantiation():
    assert kinds([{"intent": "Get the top-7 best-selling brand in March 2023"}]) == ["template"]


def test_answer_fires():
    assert "answer" in kinds([{"intent": "Name a fitness band sold in the store", "answer": ["Quest Lumaflex™ Band"]}])


def test_short_answers_ignored():
    assert kinds([{"intent": "How many reviews does the blue mug have", "answer": ["2"]}]) == []


def test_clean_intents_pass_fast_checks():
    assert kinds([{"intent": "Change the shipping address of order 000000171 to 12 Elm St, Boston"},
                  {"intent": "Upvote the newest post in the forum about tea"}]) == []


def test_refuses_empty():
    with pytest.raises(SystemExit):
        check([], TEST, embed=False)


@pytest.mark.slow
def test_embed_fires_on_paraphrase_and_not_on_unrelated():
    pytest.importorskip("sentence_transformers")
    hits = check([{"intent": "Get the top 1 best selling product names in 2022 please"},
                  {"intent": "Create a new GitLab milestone named Sprint 9 due next Friday for the dotfiles repo"}], TEST)
    assert any(h["i"] == 0 and h["kind"] in {"embed", "exact", "template"} for h in hits)
    assert any(h["i"] == 0 and h["kind"] == "embed" for h in hits), hits
    assert not any(h["i"] == 1 for h in hits), hits
