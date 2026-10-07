# Standalone pretraining export review

2026-10-07. Bounded read-only review of `pretraining_package`: four Russian
entry/card/assessment/reproduction documents, verifier/tests, as-run101 source,
selected artifacts/facts, source archives, dependency metadata and notices.
No training, model forward, Git/external write or historical source/artifact edit.

## Findings first

1. **Important redistribution completeness, fixed:** the copied upstream
   `licenses/torchgfn-LICENSE` is a13-line Apache notice/link, not full license
   text. Root retained it and added official `licenses/Apache-2.0.txt`; I verified
   its SHA against `evidence/EXPORT_SAFETY_CHECK.json`. This follows the copy
   requirement in [official Apache-2.0 §4](https://www.apache.org/licenses/LICENSE-2.0).
   Source-notice Apache versus metadata MIT remains explicitly unresolved; this
   review is not comprehensive legal clearance or a relicensing determination.
2. **Important source/validation-scope wording, fixed:** original101 ZIP members
   remain byte-identical, but five packaging-time validation files were added
   separately. README/REPRODUCE now state this, with explicit origin/SHA evidence.
   The verifier's13endpoint clock checks/three fresh final-law evaluations are
   distinct from19 included checkpoints and my previous19-clock historical audit.
3. **Low malformed-state robustness, root fix reported:** initial verifier used
   `int(step)` and could accept a fractional clock after truncation. Actual
   archived clocks are integral, so no observed artifact failure. Root replaced
   this with strict30finite/exact-value validation and reports a RED→GREEN test
   rejecting fractional/NaN/wrong ages; six package tests/lint pass. Fresh larger
   suites/weight verification for this final verifier revision remain root-owned.

**Bounded verdict:** no remaining scientific-content, needed-weight/source-byte or
import-layout blocker found. Final inventory must be regenerated after all fixes,
receipts and this review, then `verify.py` must pass against the actual release
tree. I have not certified that final manifest or GitHub publication state.

## Verified scientific content and provenance

The card separates facts from assessment and specifies all eight exact modes,
word/bit conversion, pair-distance matrix,79-point disjoint radius2 sets and dense
reward. I independently checked matrix and nearest-mode histogram
8/96/528/1600/1672/192, and reconstructed q_pre/q_R scalar/oracle quantities with
error0. Alpha12 gives exp(-d) data/teacher; environment exponent24 gives exp(-2d)
reward, not a hidden pretraining-target change.

Hard and teacher-soft losses correctly share the same extensive6/l expected
denoising objective. Labels are integrated in the soft estimator; contexts/masks
remain sampled. Uniform mask level/subset law, full-mask probability1/6, no
all63training enumeration/K4/RL, batch256 and data/mask streams are explicit.
398980-parameter/twolayer bidirectional pre-LN/learned-position architecture,
four-class output, initialization, float32 and optimizer/clipping choices match
the as-run factory/primitives, not a scaled LLaDA/author checkpoint claim.

Adam100 disclosure is correct and prominent:12500 ancestral updates, stored
clock16500;101/102 clock12500. Same-age/four-arm same-Adam causality is disclaimed;
the actual calibrated weights remain measured references, not q_pre renamed.
The selected stage artifacts include the required parent weights/hashes; absent
intermediate binaries/RL runs are explicitly not claimed included.

Independently executed only the stdlib source/archive checks:12 source ZIPs and
72 vendored framework Python hashes pass; as-run101 archive members match the
extracted source. Five added test files match both recorded SHA and original
parent files.13stage endpoints and19tensor files are present. JustGRPO/upstream
notice copies match their original local source files; foreign code is not
silently blanket-relicensed.

## Runtime, verifier and numerical evidence

Retained source layout supports original ROOT-relative pins; vendor/source
precedence and explicit model/backbone/gfn-origin checks prevent the documented
fresh CLI from silently using parent implementations. The original source needs
its declared external dependencies/eager gfn imports. Clean wheel installation,
Linux trainer portability and full12500-step/bitwise reproduction are explicitly
unverified; macOS resource monitoring is not disguised as portable training.

Verifier uses weights_only/strict model loading, source/archive hashes, finite
optimizer moments, normalized finite-law propagation, actual calibration gates
and CPU-versus-savedMPS differences. Bounds1e-4logword/1e-5scalar/1e-6terminal mass
are stated as tolerances, not observed errors. I read the actual CPU receipt:
all three final models calibrated, observed maxima3.426738e-6/1.362555e-7/
3.016913e-9 respectively. I did not run those neural evaluations. Root reports
221 parent/local/package tests and22 exported-source tests passed, including
small CPU/MPS pretraining/replay gates; subsequent strict-clock revision requires
the final reruns root is performing. None is a fresh full pretraining run.

Inventory rejects unknown/missing/mutated registered files, unsafe paths and
non-excluded symlinks; ignored Git/cache/work state is deliberately outside the
release claim. Self-hashes provide integrity, not third-party authenticity or
proof of scientific claims. The old286-file CPU receipt predates later additions
and is labeled accordingly; final inventory/readback, not that old count, rules.

No known credential-pattern hits found in inspected text or all source-ZIP
members. This is not universal secret detection; PT/NPZ were not scanned as text.
Historical absolute workstation paths and dependency/build metadata are disclosed.
`.gitignore` excludes work/venv/cache output; no external publication was performed
or independently checked. Root owns final release sealing and private-repository
visibility/access verification.
