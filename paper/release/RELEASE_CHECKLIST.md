# Release checklist

Manual steps, in order. Commands assume the repository root `/home/rockrock/cmu_paper`; the root is owned by root, so file copies there need `sudo`.

## A. Decide before publishing

1. **KVQuant patch license.** `third_party/patches/kvquant/0001`–`0004` contain upstream KVQuant code, and the upstream tree has no LICENSE file (only an "Apache Software License" classifier in `deployment/pyproject.toml`). Decided: publish. `NOTICE` states the facts (no LICENSE file upstream; Apache classifier in `pyproject.toml`). The patches were already public on `main`.
2. **vLLM file notices (Apache-2.0 §4).** The release branch already carries the vLLM LICENSE in `src/kvbench/third_party/vllm_turboquant/LICENSE`, and NOTICE lists which files were changed. Apache-2.0 §4(b) also asks for a change notice inside each modified file (`triton_turboquant_decode.py`, `triton_turboquant_store.py`, `triton_decode_attention.py`), and `compat.py` carries vLLM code without an SPDX/copyright header. Adding those comment lines does not affect behavior, but these files are among the hash-locked paths of the performance freeze. On the release branch they would then differ from the code that produced the results. Decide whether to add them; the release files in `overlay/` do not.
3. **Copyright holder.** Decided: `Copyright 2026 fzlzjerry` in `NOTICE`.
4. **Package contents.** Decide whether to publish the two CPU packages as they are. The scan found no credentials, tokens, usernames, or home paths, but found:
   - private storage URIs (`r2://kvbench-artifacts/...`): 51 occurrences in 15 files of the joint-results package, 1 in the modeling package;
   - temporary paths such as `/tmp/q2b-closeout-…`;
   - GPU UUID and PCI bus ID in `inputs/freeze_hardware_identity.json` (joint-results package);
   - internal process files `operator_prompt.md` and `execution_authorization.json` (joint-results package).

   Editing a package invalidates its checksum ledger and `COMPLETE` marker, so any redaction needs a rebuilt package with new checksums.

   Decided: publish both packages as sealed. The same GPU UUID, `r2://` URIs, and process documents are already public on `main`, and the scan found no credentials.

## B. Release branch

5. Set a Git identity for the account that owns `.git`, then build the orphan `release` branch and the `v1.0` tag. The script checks out the paths in `release_manifest.txt` from `main` (commit `0641de4b`) and applies `overlay/` (README, Makefile, LICENSE, NOTICE, the third-party LICENSE copies, and a CLI without the report subcommand). It then adds `paper/manuscript.md`, `paper/figures/`, and `paper/scripts/`, verifies every staged file against `release_files.sha256`, commits "Release v1.0", tags `v1.0`, and switches back to `main`. It never pushes and never changes `main`:

   ```bash
   cd /home/rockrock/cmu_paper
   sudo git config --global user.name "<name>"
   sudo git config --global user.email "<email>"
   sudo bash paper/release/make_release_branch.sh
   ```

   If the script stops after `Switched to a new branch 'release'`, nothing has been committed or tagged and `main` is unchanged. Return to `main` and run it again:

   ```bash
   sudo git read-tree --empty
   sudo git switch -f main
   sudo rm -f -- LICENSE NOTICE src/kvbench/third_party/vllm_turboquant/LICENSE third_party/patches/kivi/LICENSE
   sudo git status --short
   ```

   `read-tree --empty` clears the index without touching files, so `switch -f` does not delete the untracked paper files. The `rm` removes the overlay files that `main` does not track; their sources stay in `overlay/`. `git status --short` should list no tracked changes.

6. Inspect the result. The tree hash must be `0ab77c10bc66ab9249a64cd5c9caa76bda412098` (the commit hash differs between runs because it includes the commit time):

   ```bash
   sudo git rev-parse 'release^{tree}'
   sudo git log --stat -1 release | head -40
   sudo git show --no-patch v1.0
   ```

7. Rename the GitHub repository `fzlzjerry/cmu_paper` → `fzlzjerry/kvbench` (Settings → General → Repository name), then update the remote:

   ```bash
   sudo git remote set-url origin https://github.com/fzlzjerry/kvbench.git
   ```

8. Push the release branch and tag. `main` is already on GitHub at `0641de4b` and stays as it is:

   ```bash
   sudo git push origin release
   sudo git push origin v1.0
   ```

9. Make `release` the default branch (Settings → Branches → Default branch). Leave `main` published so that `0641de4b` stays reachable; the manuscript and README point to it.

## C. Release assets

10. Done: the archives are in `paper/release/assets/`, built with sorted names, numeric owner 0, and `gzip -n` so a rebuild gives the same bytes. Each unpacks to the sealed package directory unchanged (54 and 70 files):

    ```text
    466230491e6b9f1c75d25be48108b051857453c01f13a8ed643871009d0f33ff  kvbench-modeling-repro.tar.gz
    3a1aa09178c12f7fdb2bf254bb48b0761db3f6659dd2e5c8bd5f44f5c566a56b  kvbench-joint-results-repro.tar.gz
    ```

11. Create a GitHub Release from tag `v1.0` with `RELEASE_ASSETS_README.md` as the release notes, and upload the two archives, `SHA256SUMS`, and `RELEASE_ASSETS_README.md` (web: Releases → Draft a new release → choose tag `v1.0`; or `gh release create` after `gh auth login`).

## D. Verify

12. Check that each URL opens without signing in:
    - https://github.com/fzlzjerry/kvbench (default branch `release`, with README, LICENSE, and NOTICE at the root)
    - https://github.com/fzlzjerry/kvbench/releases/tag/v1.0 and the three asset downloads
    - https://github.com/fzlzjerry/kvbench/tree/0641de4b7042d46323b888ad4371a09c04373584 (full research history)
13. Confirm that the old URL `https://github.com/fzlzjerry/cmu_paper` redirects to the new one.

The earlier `git rm` route in `paper/AGENT_DOCS_AUDIT.md` is superseded by the orphan branch; `main` keeps every file.
