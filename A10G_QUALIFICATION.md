# A10G qualification of the submitted container — sanitized summary

The exact image that was submitted to the challenge Docker queue — pulled by digest, not rebuilt —
was executed on **one NVIDIA A10G with 24 GB of VRAM** (23,028 MiB reported, driver 580.159.03,
exactly one visible GPU). The image digest is recorded with the challenge submission itself and is
not published here.

**Mechanism.** The platform exposed no OCI runtime to the job, so the image's own root filesystem and
its exact `ENTRYPOINT` were executed through a rootless userspace mechanism. This establishes
**exact-image filesystem and ENTRYPOINT compatibility on A10G**, not full OCI-runtime parity: there is
no namespace isolation, and the host NVIDIA driver libraries had to be bound in explicitly, which a
real container runtime would inject.

**Inputs.** Synthetic four-modality NIfTI volumes only, generated inside the ephemeral job, spanning
three distinct folder-name prefixes with five-digit synthetic suffixes. No challenge image, label,
prediction or metadata was used or uploaded. Input immutability was proven by complete before/after
SHA-256 comparison rather than a kernel-enforced read-only mount.

| | Smoke | Benchmark |
|---|---|---|
| Cases | 2 | 12 |
| Exit code | 0 | 0 |
| Outputs validated | 2 / 2 | 12 / 12 |
| Wall time | 64.3 s | 380.6 s |
| Mean per case | 32.2 s | 31.7 s |
| Peak reserved VRAM | 2.39 GiB | 2.49 GiB |
| Output / geometry / label / hierarchy gates | all pass | all pass |
| Inputs unchanged | yes | yes |

**Gates verified:** flat output directory; output names derived dynamically from the input folder
names with no hardcoded cohort prefix; readable 3-D NIfTI; finite integer labels restricted to
`{0,1,2,3}`; ET ⊆ TC ⊆ WT; geometry identical to each synthetic source; no nested, hidden, temporary,
symlink, duplicate or zero-byte output; five distinct checkpoints in the original order; tile step
0.5; Gaussian weighting; threshold 0.5; mirroring disabled; no component filtering; no dependency or
model download observed during inference.

**Runtime projection for 451 cases:** 4.0 h from the mean, 5.0 h with a 25 % margin, and 4.0 h from
the observed 95th-percentile case time — all against the organizers' 12 h limit. This is a
**projection from synthetic inputs, not a measured full-cohort runtime**; the separate 451-case run on
an NVIDIA A40 remains the full-cohort evidence.

**Negative fixtures.** Six of seven fail closed with zero output: missing modality, duplicate
modality, unknown modality, invalid folder name, unreadable random bytes, and non-3-D input. The
seventh, non-finite input values, is **not** a required gate and did **not** fail closed: the runner
has no non-finite input contract and processed the case. That is recorded as measured behaviour; no
rejection is claimed. Output-collision behaviour is not tested and no fail-closed guarantee is
claimed for it — the official contract supplies a fresh writable `/output`.

**Not claimed:** organizer execution, hidden-test evidence, full OCI-runtime parity, identical
behaviour on real challenge cases, or non-finite input rejection.
