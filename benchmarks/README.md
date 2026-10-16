# Embodied-memory validation

Run these from the repository root after installing the package. The replay
and audit use only Python's standard library; the Ant replay needs the `sim`
extra.

```bash
python -m benchmarks.warehouse_replay --distractors 2500
python -m benchmarks.warehouse_replay --distractors 2500 --no-optimize
python -m benchmarks.mujoco_ant_replay
python -m unittest discover -s tests -v
```

## What the benchmarks measure

`warehouse_replay` is a deterministic **synthetic** patrol with a sourced cup,
table, shelf, rooms, a competing blue cup, 250 or more irrelevant observations,
a late correction, an explicit unknown, and a second robot. It checks nine
outcomes: question-to-episode retrieval, lexical precision, two-hop room
retrieval, prior knowledge,
corrected current graph, unknown and stale abstention, missing evidence, and
robot isolation. The script exits nonzero if any check fails. The exact
all-word search baseline misses the natural-language cup question; the
retriever strips common question words, tries exact content terms, and uses a
broader term match only when exact matching finds nothing. It seeds entity
and graph lookup from matched observations. This is lexical retrieval, not
semantic understanding or answer generation.

`mujoco_ant_replay` runs Gymnasium's articulated `Ant-v5` with a fixed seed.
It records 21 pose samples over 200 simulator steps, retrieves every pose and
its observation, checks a prior knowledge cursor, and restarts the memory
runtime. It measures off-loop memory query latency, not controller timing or
robot task performance.

### Observed results in this workspace

These are single-process measurements on the current development machine,
not service-level guarantees or target-computer timings. The FTS optimization
was run *after* replay import, before queries.

| Run | Correctness | Median query | Slowest query |
|---|---:|---:|---:|
| Warehouse, 258 observations, optimized | 9/9 | 2.0 ms | 4.3 ms |
| Warehouse, 2,508 observations, unoptimized | 9/9 | 24.9 ms | 84.5 ms |
| Warehouse, 2,508 observations, optimized | 9/9 | 6.0 ms | 13.0 ms |
| Warehouse, 5,008 observations, optimized | 9/9 | 9.8 ms | 25.1 ms |
| MuJoCo Ant, 200 steps, 21 samples | 21/21; cursor and restart pass | 1.4 ms | 2.0 ms |

Run `MemoryStore.optimize_search()` only after bulk import or in a stopped
robot's maintenance window. It acquires a database write lock. Normal onboard
ingestion must not call it from the action or query path. Query latency rises
with corpus size, and broad lexical fallback can be expensive. Measure p99 and
long-duration behavior on the actual robot computer.

## OpenEQA audit and honest boundary

The [official OpenEQA annotations](https://github.com/facebookresearch/open-eqa/blob/main/data/open-eqa-v0.json)
contain 1,636 questions across 152 episode histories. This workspace audited
all annotations using `openeqa_annotations.py`: 263 object-localization, 252
object-state, 240 attribute, 231 object-recognition, 220 spatial, 217
functional-reasoning, and 213 world-knowledge questions. The official JSON's
Git blob SHA is `8e163d87b6cd9cb4f37cd636a668353dc76ff14c`.

```bash
python -m benchmarks.openeqa_annotations path/to/open-eqa-v0.json
```

This is **not an OpenEQA accuracy result**. The annotations alone do not
contain the sensory history needed to ground an answer. The
[official download instructions](https://github.com/facebookresearch/open-eqa/blob/main/data/README.md)
specify roughly 12 GB for HM3D RGB frames or 62 GB for ScanNet RGB frames,
with additional dataset access and extraction requirements. Those histories
and a calibrated visual answering model were unavailable here. We did not
insert reference answers into memory or use them as evidence.

The benchmark coverage exposes the next evaluation gate: real RGB-D/audio
adapters, scene-graph construction, entity tracking, visual grounding,
question-to-query translation, active exploration, and answer scoring. The
current software proves narrower properties of an evidence-backed memory
service and a simulator adapter.
