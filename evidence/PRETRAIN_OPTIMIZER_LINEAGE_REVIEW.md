# Historical pretraining optimizer-lineage review

2026-10-07. Bounded independent audit: read archived/existing sources and19 tensor
checkpoints; inspected installed Torch2.13.0; ran only a one-parameter/three-step
MPS fixture. No neural model, full training, historical-artifact/source edit,
package creation, Git operation or external publication by this reviewer.

## Ranked verdict

1. **Verified, load-bearing scientific provenance bug:** original reference100's
   four-arm calibration continuation does not start every arm with the same Adam
   clock. Its accepted model has12500 nominal lineage updates but Adam `step`16500
   in every one of30 parameter states. References101/102 have12500 for both.
   Claims of identical age-normalized pretraining recipes, or four-arm control of
   all Adam state, must be corrected before packaging/publication.
2. **Verified/inferred impact limit:** this is a phantom optimizer-clock advance,
   not4000 extra updates to the accepted model or transplanted extra4000 moment
   updates. Mature bias correction makes the isolated fixed-moment effect small;
   that does not bound its eventual effect on weights/calibration/RL outcomes.
3. **Unaffected measured objects:** actual fitted checkpoint weights, calibrated
   p0 tables/laws and previously rederived RL endpoints remain valid as observed
   artifacts. A claim about the cause or age-matched replication of their training
   is a different claim. No mathematical evaluator/oracle-target identity is
   refuted by this optimizer-history discovery.

## Mechanism and historical source evidence

The executed `calibration_continue.py` inside
`runs/BS-calibration-diagnostic/source_snapshot.zip` loads the CPU checkpoint once
(archived line28), executes the ordered four arms (line40) and repeatedly calls
`optimizer.load_state_dict(checkpoint['optimizer'])` (line45). Models reload parent
weights, and local data/mask streams reset; the parent optimizer dictionary is
reused without a fresh disk read/deep clone. Corresponding current formatted
locations are `checks/calibration_continue.py:33`, `:55`, `:62`.

I verified the archived entrypoint SHA
`b9d680becdcd7467565b07e7295f79e39c3c4d331c2c0619fd68f0a3bfbbf3ae`
against `campaign.json`. It differs from the current file's byte hash, but their
ASTs agree: formatting explains the difference, not a repaired historical run.
My first archival check incorrectly also required current byte identity and
stopped after checkpoint inspection; the hash/AST follow-up resolved that checker
assumption. The historical archive remains authoritative and was not changed.

Installed source:
`/Users/a2mogus/build-projs/ccm-intro/projects/unified-bench/.venv-procgen/lib/python3.11/site-packages/torch/optim/optimizer.py:773`
(`_process_value_according_to_param_policy`), and `:880` (`load_state_dict`). The
loader shallow-copies the outer dictionary (`:928`) and deep-copies param groups
(`:939`), but the tensor policy returns the original `step` object when neither
capturable nor fused is enabled. Historical groups have capturable=False,
fused=None. CPU moment tensors cast to MPS are separate allocations; CPU `step`
is deliberately retained on CPU and aliases the reused dictionary. An optimizer
step increments it in place. This is a caller-state isolation failure, not proof
of a Torch implementation defect. CPU-to-CPU/same-device reuse can also alias
moments when `.to` is a no-op; the historical CPU→MPS case isolates the clock.

Independent tiny MPS reproduction: initialized a one-parameter AdamW CPU state,
set its clock6000, then loaded that same dictionary into three fresh MPS optimizers
and performed one scalar-gradient update each. Parent clocks became6001/6002/6003;
loaded `step is parent_step` was true, while parent CPU exp_avg/exp_avg_sq remained
bit-identical. Fixture exited0, with no neural model or dataset operation.

## Actual saved lineage

All ages below are uniform across all30 parameter states, independently loaded
with `weights_only=True,map_location='cpu'`. “Nominal” counts updates on this
model's ancestral path, not discarded sibling work.

| Saved stage | Nominal lineage updates | Actual Adam clock |
|---|---:|---:|
| Original100 hard6000 |6000|6000|
| Diagnostic hard-high final2000 |8000|8000|
| Diagnostic hard-low final2000 |8000|10000|
| Diagnostic soft-high final2000 |8000|12000|
| Diagnostic soft-low final2000 |8000|14000|
| Selected soft-high → extension initial |8000|12000|
| Extension final4000 |12000|16000|
| Low-LR tail initial |12000|16000|
| Accepted100 tail500 |12500|16500|
| Independent101/102 hard6000 |6000|6000|
| Independent101/102 soft-high initial/final |6000/12000|6000/12000|
| Independent101/102 soft-low initial/final |12000/12500|12000/12500|

Accepted100 checkpoint SHA remains
`788e00b95e0574716c15893a25205a1a84c0eb5ddf088d4ab392add41bc8c674`.
Extension manifest selects diagnostic **soft-high**, not soft-low; both extension
and tail parent hashes independently match their actual files. The selected
soft-high inherited +4000 from the two earlier discarded hard arms. The later
discarded soft-low arm did not retroactively change the already serialized
soft-high checkpoint: `bitseq/artifacts.py:13–19` clones CPU tensors at save.

Thus12500 actual ancestral updates and3200000 ancestral batch256 draws remain
the correct nominal recipe accounting. The known accepted100 `samples=1664000`
is soft-stage accounting, not that ancestral total. Total discarded diagnostic
work is separate from both ancestral work and the phantom-clock offset. Do not
rename16500 as16500 learned-model updates.

## What can and cannot be bounded

With the **same updated moments and gradient**, beta(.9,.999), weight decay0 and
eps approaching0, changing nominal time t to t+delta multiplies an Adam coordinate
update by

`sqrt((1-beta2^(t+delta))/(1-beta2^t)) * (1-beta1^t)/(1-beta1^(t+delta))`.

Positive epsilon reduces the second-moment effect; beta1 correction is already
saturated at6000. At the first nominal continuation update6001, the fixed-moment
upper multipliers are1.00106960/+2000,1.00121412/+4000 and1.00123366/+6000.
For the accepted +4000 lineage, later upper multipliers are1.00016388 at8001,
1.000002995 at12001 and1.000001818 at12500. This isolates a maximal~.1214% early
age-only update-scale effect for the selected branch under those assumptions.

It is **not** a bound on actual accumulated parameter distance, metric changes or
calibration acceptance. Changed parameters alter later gradients/moments;
clipping, numerical kernels and stochastic optimization can amplify perturbations.
No counterfactual corrected neural training was run. Neither “caused the measured
differences” nor “scientifically negligible” is established.

## Other stages and required claim narrowing

- Four-arm LR/estimator continuation: “same parent Adam state” is false beyond
  hard-high. Retain observational results but do not isolate LR/soft-estimator
  causality from this flawed controlled comparison. Frozen hard-versus-soft
  expected-gradient/MSE calculations in `diagnose_calibration.py` perform no
  optimizer update and are not invalidated by this clock alias.
- Original hard6000 is clock-consistent. Extension/tail use fresh disk reads via
  `artifacts.load_checkpoint:57–74` and inherit, rather than create, +4000. Saved
  hard/selected parents are not silently mutated on disk.
- Independent101/102: `checks/reproduce_base.py:158–165` validates actual parent
  clocks; `:184–194` reads/loads fresh disk records at each serial transition.
  All inspected initial/final stage clocks match nominal counts. These two bases
  implement the intended fixed schedule, not exact optimizer-history replication
  of historical100. Cross-reference outcomes remain measured independent fits,
  but reference seed alone is no longer the only pretraining-history distinction.
- RL panels/controls instantiate fresh Adam optimizers at
  `updates.py:50–77`, not the pretraining optimizer. Within a base their starting
  optimizer state/weights and first-input comparisons remain intact. Restores
  read fresh disk records; no same retained parent dictionary is shared across
  those arms. Frozen audits load policy/head weights, not pretraining moments.
  The warmed-Adam neural gates explicitly deep-copy optimizer states. This is a
  bounded reachable-path review, not a repository-wide aliasing certification.

For the Russian card/package: report both ancestral updates and actual stored
Adam clock, retain immutable original files/hashes and the clock deviation in
the reproducibility contract. “Calibration passed” and “actual fitted p0” remain
true; “same age-normalized recipe as101/102” must not survive. Do not overwrite
historical clocks to cosmetically repair provenance. Future independent branches
need fresh deserialization/deep-cloned optimizer state plus explicit initial/
final clock and moment-isolation gates; a corrected future recipe cannot promise
bitwise reproduction of the bug-affected100 artifact. Root owns any package or
code changes and external publication.
