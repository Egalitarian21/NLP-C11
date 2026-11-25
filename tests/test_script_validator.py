import copy
import json
import unittest
from pathlib import Path

from src.script_validator import validate_script


ROOT = Path(__file__).resolve().parents[1]


def p005_passage():
    data = json.loads((ROOT / "data" / "v0.3" / "dataset.json").read_text(encoding="utf-8"))
    return next(item for item in data if item["passage_id"] == "p005")


def valid_script():
    passage = p005_passage()
    return {
        "schema_version": "1.0",
        "passage_id": "p005",
        "generation_method": "direct",
        "title": "Club Adventure",
        "language": "en",
        "target_words": list(passage["target_words"]),
        "characters": [
            {"id": "c1", "name": "Mia"},
            {"id": "c2", "name": "Leo"},
        ],
        "scenes": [
            {
                "scene_id": "scene_1",
                "location": "School club fair",
                "dialogue": [
                    {
                        "line_id": "line_001",
                        "speaker_id": "c1",
                        "text": "Join our cooking club! We can cook beef noodles soon.",
                        "emotion": "excited",
                        "action": "wave",
                        "source_sentence_ids": ["s003", "s005", "s006"],
                    },
                    {
                        "line_id": "line_002",
                        "speaker_id": "c2",
                        "text": "I love books, but I also want to discover nature.",
                        "emotion": "curious",
                        "action": "point",
                        "source_sentence_ids": ["s008", "s014", "s019"],
                    },
                    {
                        "line_id": "line_003",
                        "speaker_id": "c1",
                        "text": "Then join the Nature Club. We collect plants and insects.",
                        "emotion": "happy",
                        "action": "point",
                        "source_sentence_ids": ["s003", "s017"],
                    },
                    {
                        "line_id": "line_004",
                        "speaker_id": "c2",
                        "text": "That sounds great! I want to explore with friends.",
                        "emotion": "excited",
                        "action": "nod",
                        "source_sentence_ids": [],
                    },
                ],
            }
        ],
    }


class ScriptValidatorTests(unittest.TestCase):
    def assert_invalid(self, script, fragment=None):
        result = validate_script(script, p005_passage())
        self.assertFalse(result["valid"])
        self.assertTrue(result["errors"])
        if fragment:
            self.assertTrue(any(fragment in (item["path"] + " " + item["message"]) for item in result["errors"]))
        return result

    def test_valid_mock_script_passes_schema_and_rules(self):
        result = validate_script(valid_script(), p005_passage())
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["stats"]["target_words_found"], 5)
        self.assertIn("estimated_duration_sec", result["stats"])
        self.assertTrue(result["semantic_review_required"])

    def test_wrong_passage_id_is_rejected(self):
        script = valid_script()
        script["passage_id"] = "p999"
        self.assert_invalid(script, "passage_id")

    def test_missing_target_word_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][2]["text"] = "Then join the Nature Club. We watch birds and take photos."
        self.assert_invalid(script, "Missing target word")

    def test_duplicate_line_id_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][1]["line_id"] = "line_001"
        self.assert_invalid(script, "line_id")

    def test_unknown_speaker_id_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["speaker_id"] = "c9"
        self.assert_invalid(script, "speaker_id")

    def test_illegal_emotion_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["emotion"] = "angry"
        self.assert_invalid(script, "emotion")

    def test_illegal_action_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["action"] = "run"
        self.assert_invalid(script, "action")

    def test_invalid_source_sentence_id_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["source_sentence_ids"] = ["s999"]
        self.assert_invalid(script, "source_sentence_ids")

    def test_character_limit_is_rejected(self):
        script = valid_script()
        script["characters"].append({"id": "c3", "name": "Sam"})
        script["characters"].append({"id": "c4", "name": "Kai"})
        self.assert_invalid(script, "characters")

    def test_scene_limit_is_rejected(self):
        script = valid_script()
        for i in range(2, 5):
            scene = copy.deepcopy(script["scenes"][0])
            scene["scene_id"] = f"scene_{i}"
            scene["dialogue"] = []
            script["scenes"].append(scene)
        self.assert_invalid(script, "scenes")

    def test_substring_is_not_a_valid_target_match(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["text"] = "Join our cooking clubhouse! We can cook beef noodles soon."
        script["scenes"][0]["dialogue"][2]["text"] = "Then join the Nature Clubhouse. We collect plants and insects."
        self.assert_invalid(script, "Missing target word")

    def test_reasonable_inflection_is_accepted(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][2]["text"] = "Then join the Nature Club. We are collecting plants and insects."
        result = validate_script(script, p005_passage())
        self.assertTrue(result["valid"], result)

    def test_empty_dialogue_text_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["text"] = ""
        self.assert_invalid(script, "text")

    def test_whitespace_only_dialogue_text_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"][0]["text"] = "   "
        self.assert_invalid(script, "text")

    def test_empty_scene_is_rejected(self):
        script = valid_script()
        script["scenes"][0]["dialogue"] = []
        self.assert_invalid(script, "dialogue")

    def test_p005_mock_introduces_three_clubs_and_uses_reviewed_evidence(self):
        mock = json.loads((ROOT / "examples" / "mock_script.json").read_text(encoding="utf-8"))
        lines = {
            line["line_id"]: line
            for scene in mock["scenes"]
            for line in scene["dialogue"]
        }
        text = " ".join(line["text"].casefold() for line in lines.values())
        self.assertIn("cooking club", text)
        self.assertIn("book club", text)
        self.assertIn("nature club", text)
        self.assertEqual(lines["line_001"]["source_sentence_ids"], ["s003"])
        self.assertEqual(lines["line_006"]["source_sentence_ids"], ["s014"])
        self.assertEqual(lines["line_016"]["source_sentence_ids"], [])


if __name__ == "__main__":
    unittest.main()
