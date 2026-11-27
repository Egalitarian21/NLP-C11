# C11 Skit Interface Specification

Version: 1.0

This document defines the stable handoff between NLP generation (A), TTS and
evaluation (B), and animation (C). It covers the interface only; no real LLM
script generation, TTS, or animation is performed in this phase.

## Input

NLP receives one passage object from `data/v0.3/dataset.json`, `dev.json`, or
`test.json`. Required fields are `passage_id`, `title`, `language`, `sentences`,
`target_words`, and `constraints`. `sentences` contains ordered objects with
`id` and `text`; `target_words` contains the five frozen V0.3 words and must not
be replaced by V0.2 values. `constraints` includes duration and count limits.

`target_word_details` is evidence and scoring metadata. It is not a replacement
for the target-word list and need not be copied into a script output.

## NLP output

The output is validated by `schemas/script.schema.json` and then by
`src/script_validator.py` against the input passage. The canonical complete,
independently validated example is
`examples/mock_script.json`. Use that file for interface testing; it contains
two characters, two scenes, globally unique line IDs, all five current P005
V0.3 target words, and the required evidence fields.

There is intentionally no shortened inline JSON example here. A shortened
snippet can omit required fields such as the second character and therefore
must not be copied as a standalone interface payload.

`generation_method` is exactly `direct` or `plan_then_generate`; `language` is
`en`. The preferred number of characters is 2--3 and scenes 1--3, subject to
the passage constraints. Every script must use all five target words naturally;
simply listing words is invalid.

The NLP interface intentionally has no TTS file path, timestamp, video path, or
rendering state. Those belong to downstream manifests.

## IDs and ordering

- Character IDs use `c1`, `c2`, and so on; they are unique within a script.
- Scene IDs use `scene_1`, `scene_2`, and so on; scenes are consumed in array order.
- Line IDs use `line_001`, `line_002`, and so on; they are globally unique.
- Dialogue is consumed in scene-array order, then dialogue-array order.
- `speaker_id` must refer to an existing character ID.

## Evidence references

`source_sentence_ids` refers to the `id` field of an input sentence, for example
`s017`. A line stating or paraphrasing a textbook fact should cite every
relevant source sentence. A purely creative transition may use an empty array,
but an empty array must not be used to evade fact checking. The validator checks
that IDs exist; existence alone does not prove semantic faithfulness.

## Enum values

Allowed `emotion`: `neutral`, `happy`, `sad`, `curious`, `surprised`, `excited`.

Allowed `action`: `none`, `nod`, `point`, `wave`, `look_at`, `walk`.

## Validation command

```powershell
python src\script_validator.py `
  --script examples\mock_script.json `
  --passage-id p005 `
  --dataset data\v0.3\dataset.json `
  --schema schemas\script.schema.json `
  --report outputs\interface_validation_report.md
```

The validator separates structural errors, deterministic rule errors,
non-blocking duration warnings, and semantic-review items. Its duration is an
explicit WPM estimate, never measured audio. B must replace it with TTS timing.

## Handoff to B (TTS)

1. Validate the script and preserve the ordered `(scene_id, line_id)` list.
2. Resolve `speaker_id` to a character name and voice configuration.
3. Synthesize one line or controlled utterance per `line_id`. The required
   collision-safe path convention is:
   `audio/{passage_id}/{generation_method}/{run_id}/{line_id}.wav`.
   For example:
   `audio/p005/direct/run_001/line_001.wav`.
4. Record the measured duration of that audio line as `duration_sec` in the
   downstream `audio_manifest.json`.
5. If the final video is assembled, record its timeline placement separately as
   `start_sec` and `end_sec`. These are video-timeline values, not the audio
   file's duration.
6. Do not write audio paths or timestamps back into the NLP script JSON.

`examples/mock_audio_manifest.json` is a mock only: it has a collision-safe
path pattern, null audio duration, and null video-timeline timestamps. It does
not claim that audio exists.

## Handoff to C (animation)

C consumes scene order, `location`, character IDs/names, dialogue text,
`speaker_id`, `emotion`, and `action`. It can map `action` to an animation
library and use `source_sentence_ids` for review overlays or traceability. C
should preserve line order and use `line_id` when attaching visual timing after
B supplies measured audio. Animation assets and video paths are outside this
NLP interface.

## Mock status

`examples/mock_script.json` is manually authored from the real P005 passage and
current V0.3 target list. It is an interface fixture, not a Direct Generation
result, not a Plan-then-Generate result, and not part of a formal LLM generation
comparison.
