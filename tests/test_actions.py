import pytest
from wa.actions import ParseError, parse, to_str

def test_reasoning_then_json():
    r, a = parse('I need the search box.\n{"action": "type", "id": "4", "text": "red shoes", "enter": true}')
    assert r == "I need the search box." and a == {"action": "type", "id": 4, "text": "red shoes", "enter": True}

def test_nested_stop_and_braces_in_strings():
    raw = 'Done.\n```json\n{"action":"stop","answer":{"task_type":"RETRIEVE","status":"SUCCESS","retrieved_data":["a}b", "{x"]}}\n```'
    _, a = parse(raw)
    assert a["answer"]["retrieved_data"] == ["a}b", "{x"] and a["answer"]["error_details"] is None

def test_think_block_ignored_and_last_object_wins():
    _, a = parse('<think>{"action":"go_back"}</think>first {"action":"scroll","direction":"down"} then {"action":"click","id":7}')
    assert a == {"action": "click", "id": 7}

@pytest.mark.parametrize("bad", ['no json', '{"action":"fly"}', '{"action":"click"}', '{"action":"scroll","direction":"left"}',
                                 '{"action":"stop","answer":"x"}', '{"action": "type", "id": 3}'])
def test_rejects(bad):
    with pytest.raises(ParseError):
        parse(bad)

def test_roundtrip():
    a = {"action": "select", "id": 2, "option": "Canada"}
    assert parse("x\n" + to_str(a))[1] == a
