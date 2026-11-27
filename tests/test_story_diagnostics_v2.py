import unittest
from pathlib import Path

from src.script_validator import load_passage
from src.story_diagnostics import diagnose_story
from tests.test_story_driven_v2 import make_valid_script


ROOT = Path(__file__).resolve().parents[1]


class StoryDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passage = load_passage(ROOT / "data" / "v0.3" / "dev.json", "p005")

    def test_diagnostics_are_non_blocking_and_report_target_locations(self):
        script = make_valid_script(self.passage)
        result = diagnose_story(script, self.passage)
        self.assertTrue(result["heuristic_only"])
        self.assertEqual(set(result["target_word_line_locations"]), set(self.passage["target_words"]))
        self.assertIn("line_001", result["target_word_line_locations"]["join"])

    def test_exact_source_copy_and_long_turns_are_review_flags(self):
        script = make_valid_script(self.passage)
        script["scenes"][0]["dialogue"][0]["text"] = "Join our cooking club!"
        script["scenes"][0]["dialogue"][1]["text"] = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five"
        result = diagnose_story(script, self.passage)
        self.assertIn("line_001", result["exact_source_copy_line_ids"])
        self.assertIn("line_002", result["long_informational_line_ids"])
        self.assertTrue(result["review_flags"])


if __name__ == "__main__":
    unittest.main()
