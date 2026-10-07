# Source reuse and adaptation notices

Canonical task/estimator/sampler/replay/TB code is imported from torchgfn's pinned original
Python source, not copied/modified in bitseq. Its original LICENSE is retained under
.sources/torchgfn and package identity/metadata-only build patch in SETUP/setup_manifest.
The source LICENSE says Apache-2.0 while packaging metadata says MIT; we do not resolve
that discrepancy by assumption. Existing repository model/scoring code is reused locally.

The narrowly ported JustGRPO advantage and token-ratio update are derived from
LeapLabTHU/JustGRPO, commit1a2fddb5c6655597e63081c0af5ebb718a849f39, grpo.py.
Grouped-layout/device/generation adaptations and executed-source gradient tests are declared
in PROPOSAL/PROPOSAL_REVIEW. Original notice follows:

MIT License

Copyright (c) 2026 LeapLab, Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
