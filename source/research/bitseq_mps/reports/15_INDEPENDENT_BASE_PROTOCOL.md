# Fixed amended independent bases101/102 — preparation, not results

2026-10-07. Owner of run sequencing is the receiving/root conversation. No training
or neural/device gate was launched by this preparation unit. Existing primary panel
and numerical audit have compute priority. Only stdlib contract checks ran.

Question: does the accepted amended optimization recipe calibrate independently
initialized learned q12 references? This is a prerequisite for later frozen-contrast
replication, not a new method-ranking study or a claim about8B models.

## Frozen recipe and scientific object

Keep original n12/k2/eight XOR modes, q12 proportional toexp(-d), Transformer
initialization/input/output specialization, uniform mask-level/subset law and all
four existing exact acceptance gates unchanged. For each seed101 and102:

1.6000 hard-label updates at6e-4, batch256, AdamW(.9,.999)/eps1e-8/weight_decay0,
clip1. Initial model seed is the independent seed; data RNG seed=seed+10000,
mask RNG seed=seed+20000. Source: unchanged run.run_pretraining.
2.6000 teacher-conditional updates at6e-4; carry actual learned weights, all Adam
moments/update counters, CPU data/mask generators and global CPU/MPS RNG states.
3.Exactly500 teacher-conditional updates at6e-5; carry the same state again.

The soft loss integrates masked labels under exact q12 conditional marginals,
holding sampled context andmask level fixed. Its population expected NELBO is the
same hard-label objective; gradient noise/clipping/Adam dynamics can differ. No
population-gradient training, target-q24 substitution or alternate backbone.
Source primitive equality is backed by existing test_soft_pretraining; this
wrapper's copied-step/replay gate is additionally prepared but NOT RUN.

Seed100 obtained this cumulative schedule through registered, outcome-conditioned
diagnostics/extensions. Freezing it now is an explicit amended recipe for the two
independent seeds, not retroactive success of the failed original6000-step recipe.
Archived seed100 samples=1664000 counts6500 soft-stage draws; the full lineage
cost is3200000 draws=12500×256. New receipts/checkpoints use cumulative draws.

## Observation and selection amendment

To limit stored artifacts, hard stage evaluates/saves only0 and6000; soft stages
save initial state and evaluate/save their final6000/500. This changes observation
schedule from historical500-step checkpoints, not the prescribed optimizer/data
updates. None of these evaluations stops or rescues the run. Final acceptance
uses only the actual tail500 evaluation and its exact checkpoint hash. Earlier
passing checkpoints, higher reward or closeness to threshold cannot replace it.

At that final checkpoint require both decoder TVs toq12<=.05, every radius2 basin
within20% relative error under both decoders, decoderTV<=.02, and exact denoising
excess/word<=.02. The unchanged run._evaluate computes all15625 partial boards,
CPU64 softmax of device32 logits and full4096-terminal propagation without mass
renormalization. Accepted learned p0/P0/c0 remain distinct from exactq12/Bayes.
If any final gate fails, write calibration_failed and retain evidence. Do not
add updates until it passes, relax gates or choose bases by downstream RL outcomes.

## Minimal code/reuse boundary and provenance

checks/reproduce_base.py calls the unchanged hard driver. Its short soft update
is explicitly attributed to calibrate.run_soft_continuation and reuses existing
draw_pretraining_masks, soft_pretraining_loss, FiniteGraph/task_distribution/
teacher_table, checkpoint load/save, exact evaluator, archive and resource gates.
This direct adaptation is needed because current calibrate rejects a hard parent.
There is no runtime AST rewrite or third-party/shared/active-module edit. A temporary
factory seed is overwritten by strict state_dict/optimizer/global-RNG restoration;
the independent scientific seed comes from hard initialization/data/mask streams.

Hard parent.kind/seed are established by its native manifest/config hash and a
wrapper STAGE receipt without modifying archived hard artifact metadata. Soft
parents carry explicit kind/seed/stage/recipe hash and cumulative counters. Each
transition checks SHA256, expected source/task/model/loss/gate hashes, receipt,
generators, complete Adam moments and actual Adam step count. Both initial and
final state snapshots preserve the stage boundary. RECIPE and source archive are
written before training; wrapper/protocol bytes are included, alongside existing
production sources. Final CALIBRATION lives adjacent to the soft-low checkpoint,
compatible with campaign.load_calibrated_base (which re-evaluates loaded weights).

No interrupted-stage resume CLI is asserted. An interruption marks the recipe
unfinished; soft-stage interruptions attempt a failure checkpoint, whereas the
unchanged hard driver handles Exception but not every BaseException. Restart
only into a new directory after root decides, retaining partial failed evidence.
Preparation did not independently re-read the full upstream papers; it used the
transferred scientific contract, source primitives and exact existing tests.

## Resources, evidence and next commands

Measured during preparation:22GiB disk free; runs allocated about1.257GiB at the
probe (live panel can change this). Historical native checkpoint size4,840,359B,
tail checkpoint4,840,423B. With six checkpoints total/base, four full evaluations
(hard0/final, highfinal, tailfinal), source archives and logs, infer approximately
45–55MiB/base, not a measurement of new artifacts. Reserve160MiB/base at launch;
two bases are expected well below2GiB but aggregate storage must be rechecked.
Cap accounting uses unique(device,inode) pairs and excludes symlinks, matching
campaign.check_storage; hardlinked latest/milestone files are not double-counted.
Default cap is all bitseq_mps/runs, not only this wrapper's output. Existing10GiB
free reserve,2GiB RSS, normal pressure and<=1GiB swap rise gates remain in force.
Hard driver checks native resources every25; soft loop also checks aggregate
storage/source every25. Serialized execution avoids competing neural workers.

Dated benchmark: hard pretraining MPS13.324ms/update after10 warmup updates,
full exact eval.28683s (report02). No measured new soft-stage throughput or ETA
is claimed. Do not use RL timings as pretraining measurements.

Five stdlib-only gates observed RED for absent wrapper/contracts or inode overcount,
then GREEN: independent
hard config/seed/budget refusal, parent kind/source/lineage/counter refusal, final
500-only acceptance/failure, finalization of a real-shaped stage receipt containing
status=complete, and unique-inode storage accounting. Root review caught the
duplicate-status keyword publication hazard and hardlink cap overcount before
any neural execution. These are not neural calibration evidence. Actual
neural bridge/replay gate and combined suite are prepared, unrun under current
compute-sequencing constraint. After root permits sequencing, run:

```bash
# No neural imports: source pins and fixed plan (safe during primary worker).
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 research/bitseq_mps/tests/test_independent_base_recipe_static.py
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 research/bitseq_mps/checks/reproduce_base.py plan --seed 101 --out research/bitseq_mps/runs/BS-independent-base101

# Root sequences these after current worker/audit; no training launch before gates.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 BITSEQ_BASE_REPLAY_GATE=1 \
 PYTHONPATH=research/bitseq_mps/.deps:research/bitseq_mps:src \
 /Users/a2mogus/build-projs/ccm-intro/projects/unified-bench/.venv-procgen/bin/python \
 -m pytest tests research/bitseq_mps/tests -q -p no:cacheprovider --tb=short

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
 PYTHONPATH=research/bitseq_mps/.deps:research/bitseq_mps:src \
 /Users/a2mogus/build-projs/ccm-intro/projects/unified-bench/.venv-procgen/bin/python -u \
 research/bitseq_mps/checks/reproduce_base.py run --seed 101 --device mps \
 --out research/bitseq_mps/runs/BS-independent-base101
# Then the same command with seed102 and a new BS-independent-base102 output.
```

The disposable replay gate compares real two-update hard checkpoint restoration,
then two soft-high/two soft-low updates against a separate literal loop, checking
actual parameters, Adam moments, both context RNGs and parent immutability. It
does not claim to test full12500-update calibration or the MPS noise floor.
After gates pass, root reviews the source/recipe and decides when to run101/102.
Preparation creates no primary findings, stages no files and commits nothing.
