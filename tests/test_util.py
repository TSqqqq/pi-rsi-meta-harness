import unittest

from rsi_harness.util import json_objects_with, parse_json_from_agent_text


class TestAgentJsonParsing(unittest.TestCase):
    def test_think_wrapped_json(self):
        text = '<think>internal reasoning</think>\n{"hypotheses": [{"title": "x"}]}'
        self.assertEqual(parse_json_from_agent_text(text, "hypotheses")["hypotheses"][0]["title"], "x")

    def test_fenced_trailing_comma_json(self):
        text = 'Here is the result:\n```json\n{"hypotheses": [{"title": "x",}],}\n```'
        self.assertEqual(len(parse_json_from_agent_text(text, "hypotheses")["hypotheses"]), 1)

    def test_embedded_json(self):
        text = 'Result follows: {"hypotheses": [], "note": "done"} end.'
        self.assertEqual(parse_json_from_agent_text(text, "hypotheses")["hypotheses"], [])


    def test_missing_comma_between_lines(self):
        text = '{"hypotheses": [{"title": "x",\n "mechanism": "m"\n "expected_delta": 0.1}]}'
        self.assertEqual(parse_json_from_agent_text(text, "hypotheses")["hypotheses"][0]["expected_delta"], 0.1)

    def test_missing_or_extra_bracket_is_repaired(self):
        from rsi_harness.util import balance_brackets
        # Real scout slip: a list of hint objects whose last object was never closed.
        text = '{"hypotheses": [{"title": "a", "hint": [{"f": 1}, {"f": "x ] y"\n ], "d": "z"}, {"title": "b"}]}'
        obj = parse_json_from_agent_text(text, "hypotheses")
        self.assertEqual([h["title"] for h in obj["hypotheses"]], ["a", "b"])
        self.assertEqual(obj["hypotheses"][0]["hint"][1]["f"], "x ] y")  # brackets inside strings untouched
        self.assertEqual(balance_brackets('{"a": "q\\" ]", "b": [1, {"c": 2]}'), '{"a": "q\\" ]", "b": [1, {"c": 2}]}')
        self.assertEqual(balance_brackets('{"a": [1, 2}}'), '{"a": [1, 2]}')
        self.assertEqual(balance_brackets('{"ok": [1]}'), '{"ok": [1]}')

    def test_required_key_never_returns_an_inner_object(self):
        # Unbalanced outer document: repaired as a whole, never the first hypothesis passed off as the answer.
        text = '{"hypotheses": [{"title": "a", "hypothesis": "h1"}, {"title": "b", "hypothesis": "h2"}]'
        obj = parse_json_from_agent_text(text, "hypotheses")
        self.assertEqual([o["title"] for o in obj["hypotheses"]], ["a", "b"])
        with self.assertRaises(ValueError):
            parse_json_from_agent_text('{"title": "a", "hypothesis": "h1"}', "hypotheses")
        self.assertEqual([o["title"] for o in json_objects_with(text, {"title", "hypothesis"})], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
