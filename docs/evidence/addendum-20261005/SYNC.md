# Addendum 2026-10-05: off-machine copies

- Sealed artifact: `addendum-20261005-20261005t214038463429z-7d9feac1-d0bc1a`
  (results/addendum-20261005 in full, addendum code, amendment, launch scripts,
  git log), root SHA-256 `57f8ff275095fa9fd4e7867aa1f85a9b063dceeb0ab07298306e61747b79687d`,
  4,768 objects, 1.79 GB, validated locally with scripts/r2_artifact.py.
- R2: `r2://kvbench-artifacts/kvbench/sha256/57f8ff275095fa9fd4e7867aa1f85a9b063dceeb0ab07298306e61747b79687d`
  (bucket lock "kvbench-evidence-indefinite", indefinite retention, not public).
  - Publish attempt 1 (2026-10-05T21:40Z): FAIL after a partial upload,
    HTTP 502 from R2 (`publish-<root>.json`); verify then reported the prefix
    incomplete (`verify-<root>.json`).
  - Publish attempt 2 (2026-10-06T00:22Z): PASS, 205 remaining objects
    uploaded, COMPLETE written last (`publish-attempt2-<root>.json`).
  - Verify (2026-10-06T01:08Z): PASS, every object downloaded and checked
    against the inventory and checksum ledger, no unexpected objects
    (`verify-attempt2-<root>.json`).
- GitHub: branch `addendum-20261005` (code, amendment, structured results,
  REPORT.md, FAILURES.md, these receipts). results/addendum-20261005/SHA256SUMS
  lists every file of the sealed copy.
- Local copy on the GPU host (not off-machine): /home/rockrock/kvbench-addendum-sealed/.
