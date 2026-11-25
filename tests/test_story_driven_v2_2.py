import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.source_grounded_story_planner_v2_1 import load_passage_from_dataset
from src.source_grounded_story_planner_v2_2 import validate_story_plan
from src.story_driven_generation_v2_2 import (
    build_direct_v2_2_request,
    build_planned_v2_2_request,
    build_story_planning_v2_2_request,
    default_v2_2_output_dir,
    generate_direct_v2_2,
    generate_plan_v2_2,
)
from tests.test_source_grounded_story_planner_v2_1 import valid_plan
from tests.test_story_driven_v2 import make_valid_script


ROOT = Path(__file__).resolve().parents[1]


def response(payload, response_id="v22-test", total_tokens=100):
    return {
        "content": json.dumps(payload, ensure_ascii=False),
        "id": response_id,
        "model": "deepseek-flash",
        "usage": {"prompt_tokens": total_tokens - 20, "completion_tokens": 20, "total_tokens": total_tokens},
        "actual_generation_parameters": {"temperature": 0, "thinking": {"type": "disabled"}},
    }


class StoryDrivenV22Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passage = load_passage_from_dataset(ROOT / "data" / "v0.3" / "dev.json", "p005")
        cls.plan = copy.deepcopy(valid_plan(cls.passage))
        cls.plan["plan_version"] = "2.2"
        cls.env = {"LLM_MODEL": "deepseek-flash", "LLM_BASE_URL": "https://api.deepseek.com", "LLM_API_KEY": "test-only"}

    def test_final_requests_share_exact_v22_requirements(self):
        direct = build_direct_v2_2_request(self.passage, self.env, ROOT)
        planned = build_planned_v2_2_request(self.passage, self.plan, self.env, ROOT)
        shared = (ROOT / "prompts" / "shared_story_requirements_v2_2.txt").read_text(encoding="utf-8").strip()
        self.assertIn(shared, direct["messages"][1]["content"])
        self.assertIn(shared, planned["messages"][1]["content"])
        self.assertEqual(direct["prompt_version"], "direct-generation-v2.2")
        self.assertEqual(planned["prompt_version"], "planned-script-generation-v2.2")
        self.assertIn("150-170", shared)
        self.assertIn("12-16", shared)
        self.assertIn("dialogue.text", shared)
        self.assertNotIn("VALIDATED SOURCE-GROUNDED STORY PLAN", direct["messages"][1]["content"])
        self.assertIn("VALIDATED SOURCE-GROUNDED STORY PLAN", planned["messages"][1]["content"])

    def test_planning_request_is_v22_and_contains_only_v22_plan_marker(self):
        request = build_story_planning_v2_2_request(self.passage, self.env, ROOT)
        self.assertEqual(request["prompt_version"], "source-grounded-story-planning-v2.2")
        self.assertIn("v2.2 planning schema", request["messages"][1]["content"])
        schema_text = (ROOT / "schemas" / "source_grounded_story_plan_v2_2.schema.json").read_text(encoding="utf-8")
        config_text = (ROOT / "configs" / "script_generation_v2_2.json").read_text(encoding="utf-8")
        self.assertNotIn("source_facts_to_preserve", schema_text)
        self.assertNotIn("source_facts_to_preserve", config_text)

    def test_v22_plan_schema_and_cross_references_validate(self):
        result = validate_story_plan(self.plan, self.passage)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["stats"]["story_beat_count"], 4)

    def test_output_path_is_version_isolated_for_p003_and_p005(self):
        for passage_id in ("p003", "p005"):
            self.assertEqual(
                default_v2_2_output_dir(ROOT, passage_id, "direct"),
                ROOT / "outputs" / "generation_experiments" / "v2.2" / passage_id / "direct",
            )
            self.assertNotIn("v2.1", str(default_v2_2_output_dir(ROOT, passage_id, "direct")))

    def test_short_script_separates_business_status_from_duration(self):
        script = make_valid_script(self.passage, "direct")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            result = generate_direct_v2_2(
                self.passage,
                ROOT,
                output,
                transport=lambda request: response(script, total_tokens=130),
                env=self.env,
            )
        self.assertTrue(result["valid"])
        self.assertEqual(result["status"], "duration_noncompliant")
        self.assertTrue(result["validation"]["schema_valid"])
        self.assertTrue(result["validation"]["business_rule_valid"])
        self.assertFalse(result["validation"]["duration_compliant"])
        self.assertEqual(result["validation"]["human_review_status"], "pending")

    def test_p003_is_allowed_but_test_dataset_is_not_read(self):
        p003 = load_passage_from_dataset(ROOT / "data" / "v0.3" / "dev.json", "p003")
        request = build_direct_v2_2_request(p003, self.env, ROOT)
        self.assertEqual(request["passage_id"], "p003")
        self.assertNotIn("data/v0.3/test.json", request["messages"][1]["content"])

    def test_planned_pipeline_keeps_plan_and_final_output_version_isolated(self):
        script = make_valid_script(self.passage, "plan_then_generate")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "out"
            result = generate_plan_v2_2(
                self.passage,
                ROOT,
                output,
                planning_transport=lambda request: response(self.plan, response_id="plan", total_tokens=140),
                generation_transport=lambda request: response(script, response_id="script", total_tokens=160),
                env=self.env,
            )
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["revision"], "v2.2")
            self.assertEqual(result["status"], "duration_noncompliant")
            self.assertTrue((output / "source_grounded_story_plan.json").exists())
            self.assertTrue((output / "script_planned.json").exists())
            self.assertEqual(result["stats"]["api_call_count"], 2)


if __name__ == "__main__":
    unittest.main()
