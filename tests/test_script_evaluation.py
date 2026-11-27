import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.script_evaluation import (
    anonymize_script,
    build_rater_materials,
    compute_automatic_metrics,
    make_anonymous_mapping,
    make_anonymous_bundle,
    make_evaluation_sheet_rows,
    make_rater_sheet_rows,
    summarize_double_rater_scores,
    summarize_manual_scores,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "generation_experiments" / "v1.1" / "p005"


class ScriptEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passage = next(
            item
            for item in json.loads((ROOT / "data" / "v0.3" / "dev.json").read_text(encoding="utf-8"))
            if item["passage_id"] == "p005"
        )
        cls.direct = json.loads((OUTPUT / "direct" / "script_direct.json").read_text(encoding="utf-8"))
        cls.direct_log = json.loads((OUTPUT / "direct" / "generation_log.json").read_text(encoding="utf-8"))
        cls.planned = json.loads((OUTPUT / "plan_then_generate" / "script_planned.json").read_text(encoding="utf-8"))
        cls.planned_log = json.loads((OUTPUT / "plan_then_generate" / "generation_log.json").read_text(encoding="utf-8"))

    def test_schema_and_duration_are_reported_as_separate_metrics(self):
        result = compute_automatic_metrics(self.direct, self.passage, self.direct_log, ROOT)
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["duration_compliance"])
        self.assertTrue(result["duration_is_estimated"])
        self.assertEqual(result["total_words"], 118)
        self.assertEqual(result["vocabulary_coverage_rate"], 1.0)

    def test_word_forms_include_reproducible_line_positions(self):
        result = compute_automatic_metrics(self.direct, self.passage, self.direct_log, ROOT)
        self.assertEqual(result["target_word_forms"]["join"], ["join"])
        self.assertEqual(result["target_word_positions"]["join"], ["line_3", "line_5"])
        self.assertEqual(result["target_word_positions"]["collect"], ["line_10"])
        self.assertTrue(result["source_reference_valid"])

    def test_anonymization_removes_method_identity_but_preserves_script(self):
        anonymized = anonymize_script(self.direct)
        serialized = json.dumps(anonymized, ensure_ascii=False).lower()
        self.assertNotIn("generation_method", anonymized)
        self.assertNotIn("direct generation", serialized)
        self.assertNotIn("plan-then-generate", serialized)
        self.assertEqual(anonymized["target_words"], self.passage["target_words"])
        self.assertEqual(len(anonymized["scenes"]), len(self.direct["scenes"]))

    def test_anonymous_bundle_has_no_method_identity(self):
        bundle = make_anonymous_bundle(
            self.passage,
            {"script_a": self.direct, "script_b": self.planned},
        )
        serialized = json.dumps(bundle, ensure_ascii=False).casefold()
        self.assertNotIn("direct generation", serialized)
        self.assertNotIn("plan-then-generate", serialized)
        self.assertNotIn("generation_method", serialized)

    def test_anonymous_mapping_is_seeded_and_contains_both_methods_once(self):
        first = make_anonymous_mapping(42)
        second = make_anonymous_mapping(42)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"direct", "plan_then_generate"})
        self.assertNotEqual(set(first.keys()), set(first.values()))

    def test_manual_summary_warns_about_out_of_range_score(self):
        rows = make_evaluation_sheet_rows("script_a", self.passage)
        rows[1]["vocabulary_context_quality_0_3"] = "4"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_manual_scores(path, self.passage["passage_id"])
        self.assertTrue(any("out-of-range" in item for item in summary["warnings"]))

    def test_empty_manual_sheet_produces_null_scores_and_missing_warning(self):
        rows = make_evaluation_sheet_rows("script_a", self.passage)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "evaluation_sheet.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_manual_scores(path, self.passage["passage_id"])
        self.assertIsNone(summary["scripts"]["script_a"]["vocabulary_context_quality_mean"])
        self.assertTrue(any("missing" in item.lower() for item in summary["warnings"]))

    def test_manual_summary_uses_passage_as_analysis_unit(self):
        rows = make_evaluation_sheet_rows("script_a", self.passage)
        for row in rows:
            if row["score_scope"] == "word":
                row["vocabulary_context_quality_0_3"] = "2"
            else:
                row["factual_faithfulness_0_5"] = "4"
                row["dialogue_naturalness_0_5"] = "3"
                row["grade_level_suitability_0_5"] = "5"
                row["story_coherence_0_5"] = "4"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "filled.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            summary = summarize_manual_scores(path, self.passage["passage_id"])
        script = summary["scripts"]["script_a"]
        self.assertEqual(script["vocabulary_context_quality_mean"], 2.0)
        self.assertEqual(script["factual_faithfulness_0_5"], 4)
        self.assertEqual(summary["analysis_unit"], "passage_id")
        self.assertEqual(summary["passage_count"], 1)

    def test_rater_sheet_has_rater_id_and_blank_subjective_scores(self):
        rows = make_rater_sheet_rows("script_a", self.passage, "rater_1")
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["rater_id"] == "rater_1" for row in rows))
        self.assertTrue(all(row["vocabulary_context_quality_0_3"] == "" for row in rows))
        self.assertTrue(all(row["factual_faithfulness_0_5"] == "" for row in rows))

    def test_double_rater_summary_keeps_missing_scores_and_reports_agreement(self):
        rows = make_rater_sheet_rows("script_a", self.passage, "rater_1") + make_rater_sheet_rows(
            "script_b", self.passage, "rater_1"
        )
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            rater_1 = temp_path / "rater_1.csv"
            rater_2 = temp_path / "rater_2.csv"
            for path, rater_id in ((rater_1, "rater_1"), (rater_2, "rater_2")):
                current = [dict(row, rater_id=rater_id) for row in rows]
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=current[0].keys())
                    writer.writeheader()
                    writer.writerows(current)
            mapping = temp_path / "mapping.json"
            mapping.write_text(
                json.dumps({"passages": {"p005": {"script_a": "direct", "script_b": "plan_then_generate"}}}),
                encoding="utf-8",
            )
            summary = summarize_double_rater_scores(
                {"rater_1": rater_1, "rater_2": rater_2}, mapping, passage_ids=["p005"]
            )
        self.assertIsNone(summary["raters"]["rater_1"]["passages"]["p005"]["script_a"]["vocabulary_context_quality_mean"])
        self.assertTrue(any("missing" in warning.lower() for warning in summary["warnings"]))
        self.assertEqual(summary["agreement"]["overall"]["comparable_count"], 0)

    def test_double_rater_summary_validates_scores_and_computes_pair_and_agreement(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            paths = {}
            for rater_id, offset in (("rater_1", 0), ("rater_2", 1)):
                rows = make_rater_sheet_rows("script_a", self.passage, rater_id) + make_rater_sheet_rows(
                    "script_b", self.passage, rater_id
                )
                for row in rows:
                    if row["score_scope"] == "word":
                        row["vocabulary_context_quality_0_3"] = str(2 + offset if offset == 0 else 3)
                    else:
                        row["factual_faithfulness_0_5"] = str(4 + offset)
                        row["dialogue_naturalness_0_5"] = "3"
                        row["grade_level_suitability_0_5"] = "4"
                        row["story_coherence_0_5"] = "5"
                path = temp_path / f"{rater_id}.csv"
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                paths[rater_id] = path
            mapping = temp_path / "mapping.json"
            mapping.write_text(
                json.dumps({"passages": {"p005": {"script_a": "direct", "script_b": "plan_then_generate"}}}),
                encoding="utf-8",
            )
            summary = summarize_double_rater_scores(paths, mapping, passage_ids=["p005"])
        self.assertEqual(summary["raters"]["rater_1"]["passages"]["p005"]["script_a"]["vocabulary_context_quality_mean"], 2.0)
        self.assertEqual(summary["raters"]["rater_2"]["passages"]["p005"]["script_a"]["vocabulary_context_quality_mean"], 3.0)
        self.assertEqual(summary["paired_comparison"]["p005"]["rater_1"]["direct_minus_plan"]["vocabulary_context_quality_mean"], 0.0)
        self.assertEqual(summary["agreement"]["by_metric"]["vocabulary_context_quality_0_3"]["comparable_count"], 10)
        self.assertLess(summary["agreement"]["by_metric"]["vocabulary_context_quality_0_3"]["percent_agreement"], 1.0)

    def test_rater_material_builder_creates_two_blind_material_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_rater_materials(ROOT, Path(temp))
            self.assertEqual(set(manifest["raters"]), {"rater_1", "rater_2"})
            for rater_id in manifest["raters"]:
                material = Path(temp) / rater_id
                payload = json.loads((material / f"dev_anonymous_evaluation_{rater_id}.json").read_text(encoding="utf-8"))
                serialized = json.dumps(payload, ensure_ascii=False).casefold()
                self.assertNotIn("generation_method", serialized)
                self.assertNotIn("plan_then_generate", serialized)
                self.assertTrue((material / f"dev_evaluation_sheet_{rater_id}.csv").exists())
            self.assertTrue((Path(temp) / "evaluation_rubric.md").exists())
            self.assertTrue((Path(temp) / "rating_instructions.md").exists())

    def test_double_rater_summary_detects_deleted_rows_against_expected_words(self):
        rows = make_rater_sheet_rows("script_a", self.passage, "rater_1")
        rows = [row for row in rows if row["target_word"] != self.passage["target_words"][0]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rater_1.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            mapping = Path(temp) / "mapping.json"
            mapping.write_text(json.dumps({"passages": {"p005": {}}}), encoding="utf-8")
            summary = summarize_double_rater_scores(
                {"rater_1": path}, mapping, passage_ids=["p005"],
                expected_words_by_passage={"p005": self.passage["target_words"]},
            )
        self.assertTrue(any(self.passage["target_words"][0] in warning for warning in summary["warnings"]))


if __name__ == "__main__":
    unittest.main()
