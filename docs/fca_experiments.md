# FCA-VPP2 experimental implementation

This patch adds a model-agnostic **Future-Contract Alignment (FCA)** layer on top of VPP2. The official VPP2 model, checkpoints, and benchmark protocols are unchanged.

## Method

The upper planner emits a typed subtask contract containing object/part/actor bindings, a goal and terminal predicate, required event order, manner constraints, forbidden conditions, and optional duration. VPP2 can generate future/action chunks conditioned on the subtask. The FCA monitor consumes predicted future states and the real simulator/robot state after each chunk. It returns `EXECUTE`, `ADVANCE`, `REPLAN`, `RETRY`, `ABORT`, or `ASK`; the controller advances only after the terminal predicate is satisfied.

The selective-imagination ablation supports `always`, `never`, and `selective` modes. In selective mode, future imagination is requested only when the monitor is uncertain or finds a contract violation.

## Files

- `src/vpp2/fca/contracts.py`: JSONL contract and event normalization.
- `src/vpp2/fca/monitor.py`: predicate registry, predictive-vs-observed monitor, and controller.
- `src/vpp2/fca/metrics.py`: intent, execution, atomic, order, temporal, violation, false-advance, recovery and prediction/action consistency metrics.
- `src/vpp2/fca/heads.py`: lightweight temporal contract head for cached VPP2 video features.
- `src/vpp2/fca/data.py`: JSONL dataset/collator for cached `[T,D]` features.
- `src/vpp2/fca/train.py`: frozen-backbone head training with a validation split.
- `src/vpp2/fca/rollout.py`: action-chunk gate that can wrap an existing VPP2 sampler.
- `src/vpp2/fca/counterfactual.py`: minimal-pair manifest generation.
- `src/vpp2/fca/cli.py`: command-line evaluation and manifest tools.
- `configs/fca/fca_default.yaml`: baselines and thresholds.

## Rollout JSONL format

Each line can contain:

```json
{
  "variant": "fca",
  "scene_id": "scene_001",
  "predicted_contract": {"object": "left_cup", "goal": "inside"},
  "gold_contract": {"object": "left_cup", "goal": "inside", "required_order": ["grasp", "place"]},
  "predicted_events": [{"name": "grasp", "object": "left_cup"}],
  "observed_events": [{"name": "grasp", "object": "left_cup", "t_start": 0, "t_end": 1}],
  "terminal_reached": true,
  "advanced_before_terminal": false,
  "violations": [],
  "recovery_success": true
}
```

## Commands

From the repository root:

```bash
python -m vpp2.fca validate-contracts contracts.jsonl
python -m vpp2.fca evaluate-rollouts --rollouts rollouts.jsonl --output results/fca
python scripts/evaluate_fca.py --rollouts rollouts.jsonl --output results/fca
python -m vpp2.fca generate-counterfactual cases.json cases.jsonl
python scripts/train_fca_head.py --manifest features.jsonl --output checkpoints/fca_head.pt \
  --input-dim 1024 --epochs 5 --device cuda
```

### Cached-head manifest

Each line in `features.jsonl` is one rollout window.  `features` can be an
inline nested list or a path to a CPU `.pt`/`.npy` tensor with shape `[T,D]`:

```json
{"id":"episode-0001","features_path":"features/episode-0001.pt",
 "phase":[0,0,1,2],"terminal":[0,0,0,1],"violation":[0,0,0,0]}
```

The contract head is deliberately separate from VPP2.  Cache the video/future
features once, train this head, and then load it in a deployment wrapper.  A
minimal online integration is:

```python
from vpp2.fca import Contract, FCAActionAdapter, FCAController

controller = FCAController([Contract.from_dict(contract)], max_replans=3)
adapter = FCAActionAdapter(controller, execute_chunk_steps=8, min_chunk_steps=1,
                           imagination_mode="selective")
gate = adapter.observe(observed_state, predicted_states=predicted_states)
if gate.execute:
    action_chunk = adapter.trim_chunk(vpp2_policy.infer_action(...)["action"], gate.chunk_steps)
    robot.execute(action_chunk)
```

The adapter does not alter VPP2 checkpoint loading or the official benchmark
path.  It only decides whether to execute a short action chunk, trigger future
imagination, advance to the next contract, retry, or replan.  Use
`always`, `never`, and `selective` imagination modes as the required ablation.

The evaluator writes `metrics.json` and `per_rollout.jsonl`. `contract_success` is gated by terminal success, zero forbidden violations, and complete event order, so a correct final pose cannot hide instruction violations.

## Recommended experiment matrix

Compare `vpp2_full_instruction`, `vpp2_subtask_feedback`, `fca`, `fca_no_future`, and `fca_no_contract`. Build same-scene minimal pairs that change only target object, arm, event order, approach direction, manner, or forbidden object. Report goal SR together with planner intent, lower-level execution, atomic F1, order score, temporal IoU, false advance rate, violation rate, recovery success, and prediction-action consistency. Keep official LIBERO/RoboDojo evaluation unchanged and run this layer as an additional diagnostic or gated backend.
