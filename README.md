# C11 Task A — Team Handoff

This handoff contains the shareable implementation of Task A: turning textbook-passage inputs into short English skits with a stable JSON interface, Direct Generation, and source-grounded Plan-then-Generate support.

## Team roles

- **A — NLP generation:** maintains the schemas, prompts, deterministic validation, source-evidence checks, and generation code.
- **B — TTS:** reads the validated script JSON, maps `speaker_id` to voices, and produces one audio item per `line_id`.
- **C — animation/video:** reads scene order, locations, speaker identity, emotion, action, and line order to assemble visuals.

## What is included

- The shared script schema and v2.2 source-grounded story-plan schema.
- v2.2 Direct and Plan-then-Generate prompts and the shared configuration.
- The validator, offline evaluation helpers, source-evidence diagnostics, and the v2.2 generation entry point.
- `docs/INTERFACE_SPEC.md` and `docs/TASK_A_HANDOFF.md`.
- Four development-set example scripts for P003 and P005:
  `examples/task_a_v2_2_dev/`.
- The manually authored interface fixture and mock audio manifest in `examples/`.

The four v2.2 examples are copied from saved development-set generation results. Their dialogue, character names, scene order, actions, emotions, and `source_sentence_ids` are unchanged. They are examples for downstream integration, not a complete benchmark and not a license to treat one method as superior.

## Important data boundary

This handoff intentionally does **not** include the V0.3 dataset, textbook PDFs, full textbook passages, test-set results, API request/response logs, human-review data, or frozen experiment outputs. The generation code expects the private project dataset and local API configuration; without those private inputs it is not a standalone reproduction command. B and C can use the included example scripts without receiving the textbook corpus.

Create local API settings only when needed by an authorized developer, using `.env.example` as a template. Never commit `.env`, API keys, authorization headers, virtual environments, caches, or generated output directories.

## Stable interface

The canonical fields are:

- `passage_id`, `schema_version`, `generation_method`, `title`, `language`, `target_words`
- `characters[]`: `id`, `name`
- `scenes[]`: `scene_id`, `location`, `dialogue[]`
- `dialogue[]`: `line_id`, `speaker_id`, `text`, `emotion`, `action`, `source_sentence_ids`

`scenes` and `dialogue` are consumed in array order. `line_id` is globally unique inside a script, and `speaker_id` must resolve to a character ID. The allowed `emotion` and `action` values are defined in `schemas/script.schema.json`.

The validator's 130-WPM duration is an estimate only. B must measure actual audio duration after TTS; audio duration and video timeline timestamps do not belong in the NLP script JSON.

## License and source-material note

Keep the repository's existing LICENSE and history when applying this handoff. The project code and documentation can follow that repository license. The textbook source and any derivative use of textbook wording remain subject to the course/institution's rights and are intentionally excluded from this handoff.

## Version boundary

The included examples are development-set v2.2 artifacts. They are not the ten-article test set, not v1.1/v2/v2.1 artifacts, and not human-review scores. Do not use them as hidden evaluation references or overwrite them with later TTS/video derivatives.
