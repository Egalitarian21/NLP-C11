# Task A Handoff for B and C

This document describes how to consume the validated Task A script interface. It does not add fields to the interface defined in `docs/INTERFACE_SPEC.md`.

## 1. Recommended starting example

Start with:

`examples/task_a_v2_2_dev/p005_direct_v2.2.json`

Then compare it with:

`examples/task_a_v2_2_dev/p005_plan_then_generate_v2.2.json`

The P003 pair is also included for a second development-set example. These are saved v2.2 development results, not new generation requests and not production-quality guarantees.

## 2. Reading a script

Consume the arrays in this order:

```python
for scene in script["scenes"]:
    for line in scene["dialogue"]:
        # process line in this exact order
        ...
```

Use `scene_id` and `line_id` as stable identifiers. Do not sort scenes or lines by display text, speaker name, or source sentence ID.

## 3. Resolving speakers

Build a lookup from `characters`:

```python
characters = {item["id"]: item["name"] for item in script["characters"]}
speaker_name = characters[line["speaker_id"]]
```

`speaker_id` is the machine-readable identity. The character name is display text. A missing speaker ID is an interface error and should stop downstream processing rather than being guessed.

## 4. Inputs for B (TTS)

For each dialogue line, use:

- `speaker_id` and the resolved character name to choose a voice.
- `text` as the exact speech content; do not silently rewrite it.
- `emotion` as the requested expressive hint.
- `line_id` as the stable audio key.

Recommended collision-safe output path:

`audio/{passage_id}/{generation_method}/{run_id}/{line_id}.wav`

For example:

`audio/p005/direct/run_001/line_001.wav`

After synthesis, B should create an `audio_manifest.json` containing the measured `duration_sec` for each line. The manifest is downstream output and must not be written back into the NLP script.

## 5. Inputs for C (animation/video)

For each scene, use `location` as the scene setting. For each line, use:

- `speaker_id` and character name for the on-screen actor.
- `text` for subtitles or speech alignment.
- `action` for the animation cue.
- `emotion` for facial/voice presentation.
- `line_id` to attach measured audio and later timeline placement.

Preserve scene order, then dialogue order. A downstream video timeline may add `start_sec` and `end_sec`, but those values are not part of this script interface.

## 6. Audio manifest shape

Use `examples/mock_audio_manifest.json` only as a shape example. It is explicitly mock data: it does not claim that any audio file exists. The measured audio `duration_sec` and the video timeline `start_sec`/`end_sec` are separate concepts.

## 7. Source evidence

`source_sentence_ids` refers to sentence IDs in the private input passage. It is traceability metadata, not a guarantee that a line is semantically faithful. Keep the field unchanged when copying a script into downstream processing. A creative transition may have an empty list; do not invent source IDs to make it look factual.

## 8. Original experiment vs. downstream production

The files under `examples/task_a_v2_2_dev/` are original saved experiment artifacts. B/C should preserve them as read-only inputs. Put TTS outputs, animation renders, corrected subtitles, or manually edited versions in a separate run-specific directory, for example:

`production/{passage_id}/{generation_method}/{run_id}/`

Do not overwrite the example JSON, change its dialogue, or change its IDs. If a human-approved edit is required, create a new version and record the change outside the original experiment artifact.
