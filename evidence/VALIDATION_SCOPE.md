# Verification scope — 2026-10-07

Root fresh execution, no new scientific12500-step training:

**Final post-fix execution:**222passed/162known warnings/33subtests90.22s for
parent+local+package;23passed/10warnings/11subtests7.01s for exported sources.
Six new package integrity/clock tests pass. Fresh strict-clock CPU reevaluation
again passes all3calibrations with the same error maxima below. Final logs and
VERIFICATION.json are included; preliminary counts below are dated process evidence.

- Package integrity tests5RED(missingverifier)→5GREEN; actual full export source
  verification first failed because diagnostic entrypoint was separately hashed
  outside source.files. Checker corrected to validate both original source.hash
  and independently recorded entrypoint_sha256. Historical source unchanged.
- CPU actual exported weights:3accepted models reloaded;13stage-end Adam clocks
  inspected;12source archives verified;72framework Python hashes checked.
  Final calibration passes all3; max CPU-versus-savedMPS logword error3.42674e-6,
  scalar1.36255e-7,terminal mass3.01691e-9. VERIFICATION.json stores full numbers.
  Bounds1e-4/1e-5/1e-6 are admission tolerances, not observed errors.
- Independent lineage reviewer checked19end+initial checkpoint clocks and source
  alias mechanism before package creation. This is distinct from root verifier's
 13endpoint clock loop, not19fresh neural reevaluations.
- Root full parent+local+package suite:221passed,162known warnings,33subtests,
 82.22s,exit0. All5local opt-in neural gates enabled. It uses current parent/local
  code, not solely archived package code; it is not claimed as standalone portability.
- Exported-source scoped suite:22passed,10warnings,11subtests,5.75s,exit0;
  actual CPU/MPS pretrain gradient comparison, few-update replay, checkpoint
  isolation and finite-evaluator brute references. Existing dependency overlay,
  but export bitseq/tdlm/gfn. Not fresh dependency install/full retraining/Linux.
- Source-pinned independent101 plan exits0; no workdir/model created by that plan.
- Top-level Russian docs links checked; known token/private-key patterns absent
  in text and source ZIP members. Tensor/PT/NPZ not scanned as text; no universal
  guarantee of secret detection. Files copied from explicit source/artifact list.

Original upstream shortApache notice retained; full official license added as
separate file after independent review. MetadataMIT discrepancy remains explicit.
CPU/MPS warning about norm_first/nested tensors is observed, not a silently ignored
failed numerical assertion. No optimizer/history/artifact correction concealed.

Final inventory count is read from current inventory/verification stdout; the
the earlier286-file receipt and current298-file neural receipt precede later
documentation/review/log additions, not assertions of the final release file count.
Re-run verify.py against final tree for integrity. Scientific source/weights did not
change between numerical verification and final release sealing.

Independent export review also found that int(step) could truncate a fractional
clock in deliberately malformed/rehashed data. Actual recorded clocks are integral.
Root added strict finite/equal/integer30-clock validation, one additional RED→GREEN
test rejecting16500.5/NaN/wrongage, and reran verification/suites. No artifact or
as-run source edit. Final counts/results are in the final logs/receipt, while
221/22 above document the preceding stage, not the final post-fix suite.

Release whitespace check: full `git diff --cached --check` reports two existing
formatting warnings in copied vendor/torchgfn-2.4.1.dist-info/METADATA (space-before-
tab in upstream README markup and trailing blank line). That full check exits2,
not green. Metadata bytes are retained; no installed/source scientific code is
changed for cosmetic cleanliness. The same check excluding that single copied
metadata file passes. Runtime/integrity/tests remain separately verified.
