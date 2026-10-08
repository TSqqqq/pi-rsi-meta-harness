import unittest

from rsi_harness.util import parse_json_from_agent_text


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


if __name__ == "__main__":
    unittest.main()
