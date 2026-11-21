# Study provenance and reproducibility limits

The records below identify **representative repository snapshots with matching recorded source-file hashes**. They are not proof of the exact Git commit, a clean working tree, every transitive dependency, the original data custodian's identity, or the complete execution environment at run time. When comparing Git blobs with the recorded Windows files, both LF and CRLF line endings were considered. The original run did not record a complete Git/environment attestation, so one cannot be reconstructed retrospectively.

| Study | Representative matching commit | Evidence and scope |
|---|---|---|
| Original development training and one-time internal evaluation | `0dca9dca00167f4d80f9a0c47120972b7868edb8` | The named training files match `reports/final_fit.json` and the named evaluation files match `reports/test_results.json`; other commits can contain the same versions. |
| Extended development-only comparison | `1f6e62505435e66492e9278496a1b593c6f1090a` | The named training files match the locally retained `outputs/search/deep_development/search.json`; the aggregated results and hashes are in `reports/deep_development_results.json`. |

The [artifact manifest](../reports/artifact_manifest.json) records the original study's selected input, source and artifact hashes. The extended study's [protocol](../reports/deep_development_protocol.md) and [results](../reports/deep_development_results.md) describe its configuration and results. The recorded GPU package versions are in `requirements-gpu.lock`. This file has version pins but no package hashes; it does not prove the identity of installed wheels or guarantee deterministic GPU execution. The current checkout does not distribute the local study's predictions, weights or ledger; however, older Git commits contain dataset images. See the [image-handling guidance](data_privacy.md).

Later source changes intentionally require stronger grouping, content, output-ledger and inference checks for future runs. Historical candidate selections and checkpoints do not acquire those guarantees retroactively, and the newer final-fit code must not be used to silently mix them with changed training code. Historical checkpoints require explicit legacy acknowledgement in the demo. The old holdout is already consumed and must not be evaluated again for model or threshold selection.
