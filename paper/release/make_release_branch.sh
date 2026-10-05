#!/usr/bin/env bash
# Build the orphan `release` branch and the v1.0 tag from main plus the prepared overlay.
#
# Run from the repository root, on main, with the account that owns .git
# (for example: sudo bash paper/release/make_release_branch.sh).
# The script never pushes, never modifies main, and never deletes untracked files.
set -euo pipefail

SOURCE_REF="main"
SOURCE_COMMIT="0641de4b7042d46323b888ad4371a09c04373584"
RELEASE_BRANCH="release"
RELEASE_TAG="v1.0"
COMMIT_MESSAGE="Release v1.0"

MANIFEST="paper/release/release_manifest.txt"     # paths taken from main (one per line)
OVERLAY="paper/release/overlay"                    # files added or replaced on the release branch
CHECKSUMS="paper/release/release_files.sha256"     # expected content of the final release tree
PAPER_FILES=(paper/manuscript.md paper/figures paper/scripts)

# --- 0. Preconditions ---------------------------------------------------------------
# A separate assignment makes set -e stop here if git refuses the repository
# (for example "detected dubious ownership").
TOPLEVEL="$(git rev-parse --show-toplevel)"
cd "$TOPLEVEL"
test "$(git rev-parse --abbrev-ref HEAD)" = "$SOURCE_REF" || { echo "check out $SOURCE_REF first" >&2; exit 1; }
test "$(git rev-parse "$SOURCE_REF")" = "$SOURCE_COMMIT" || { echo "$SOURCE_REF is not at $SOURCE_COMMIT" >&2; exit 1; }
test -z "$(git status --porcelain --untracked-files=no)" || { echo "tracked files have local changes" >&2; exit 1; }
! git rev-parse --verify --quiet "refs/heads/$RELEASE_BRANCH" >/dev/null || { echo "branch $RELEASE_BRANCH already exists" >&2; exit 1; }
! git rev-parse --verify --quiet "refs/tags/$RELEASE_TAG" >/dev/null || { echo "tag $RELEASE_TAG already exists" >&2; exit 1; }
for path in "$MANIFEST" "$OVERLAY" "$CHECKSUMS" "${PAPER_FILES[@]}"; do
    test -e "$path" || { echo "missing $path" >&2; exit 1; }
done

# --- 1. Copy the release inputs outside the working tree ----------------------------
# paper/ is untracked on main; keeping a copy guarantees the inputs survive the branch switch.
STAGE="$(mktemp -d)"
SWITCHED=0
on_exit() {
    local status=$?
    rm -rf -- "$STAGE"
    if [ "$status" -ne 0 ] && [ "$SWITCHED" -eq 1 ]; then
        echo "stopped on the unborn branch $RELEASE_BRANCH; nothing was committed or tagged." >&2
        echo "the paper files are still in paper/; see RELEASE_CHECKLIST.md to return to $SOURCE_REF." >&2
    fi
}
trap on_exit EXIT
cp -a "$MANIFEST" "$STAGE/manifest.txt"
cp -a "$CHECKSUMS" "$STAGE/checksums.sha256"
cp -a "$OVERLAY" "$STAGE/overlay"
mkdir -p "$STAGE/paper"
cp -a "${PAPER_FILES[@]}" "$STAGE/paper/"
( cd "$STAGE/overlay" && find . -type f | sed 's#^\./##' ) > "$STAGE/overlay_files.txt"
( cd "$STAGE" && find paper -type f ) > "$STAGE/paper_files.txt"
stat -c '%u:%g %a %n' -- . paper "${PAPER_FILES[@]}" > "$STAGE/paper_modes_before.txt"

# Copy listed files one at a time. `cp -a src/. dst` would also give dst the owner and mode
# of src, which changes directories such as the repository root; existing directories must
# keep their owner and mode.
copy_listed_files() {  # <source root> <file list>
    local path
    while IFS= read -r path; do
        mkdir -p -- "$(dirname -- "$path")"
        cp -p --remove-destination -- "$1/$path" "$path"
    done < "$2"
}

# Owner and mode of the working-tree directories that receive copied files (and of the root).
directory_modes() {
    cat "$STAGE/overlay_files.txt" "$STAGE/paper_files.txt" | while IFS= read -r path; do
        while path="$(dirname -- "$path")"; [ "$path" != "." ]; do echo "$path"; done
    done | LC_ALL=C sort -u | while IFS= read -r dir; do
        if [ -d "$dir" ]; then stat -c '%u:%g %a %n' -- "$dir"; fi
    done
    stat -c '%u:%g %a %n' -- .
}

# --- 2. Create the orphan branch and clear the index --------------------------------
# `git switch --orphan` removes tracked files from the working tree and leaves the index
# empty. Untracked and ignored files (artifacts/, local environments, paper/) are kept and
# are never added below.
git switch --orphan "$RELEASE_BRANCH"
SWITCHED=1
git rm -r -q --cached --ignore-unmatch -- . >/dev/null
test -z "$(git ls-files)" || { echo "index is not empty after the orphan switch" >&2; exit 1; }

# --- 3. Check out the release file list from main -----------------------------------
GIT_LITERAL_PATHSPECS=1 git checkout "$SOURCE_COMMIT" --pathspec-from-file="$STAGE/manifest.txt"

# --- 4. Apply the overlay and the paper files ---------------------------------------
directory_modes > "$STAGE/directories_before.txt"
copy_listed_files "$STAGE/overlay" "$STAGE/overlay_files.txt"
copy_listed_files "$STAGE" "$STAGE/paper_files.txt"
directory_modes > "$STAGE/directories_after.txt"
diff -u "$STAGE/directories_before.txt" "$STAGE/directories_after.txt" \
    || { echo "a directory changed owner or mode while copying" >&2; exit 1; }

# --- 5. Stage exactly the release tree ----------------------------------------------
GIT_LITERAL_PATHSPECS=1 git add --pathspec-from-file="$STAGE/overlay_files.txt"
GIT_LITERAL_PATHSPECS=1 git add --pathspec-from-file="$STAGE/paper_files.txt"

# --- 6. Verify the staged tree against the checked release tree ---------------------
git ls-files | LC_ALL=C sort > "$STAGE/staged.txt"
awk '{print substr($0, index($0, "  ") + 2)}' "$STAGE/checksums.sha256" | LC_ALL=C sort > "$STAGE/expected.txt"
diff -u "$STAGE/expected.txt" "$STAGE/staged.txt" || { echo "staged file list differs from the checked release tree" >&2; exit 1; }
sha256sum --check --quiet "$STAGE/checksums.sha256" || { echo "file contents differ from the checked release tree" >&2; exit 1; }
echo "staged $(wc -l < "$STAGE/staged.txt") files; contents match $CHECKSUMS"

# --- 7. Commit and tag ---------------------------------------------------------------
git commit -q -m "$COMMIT_MESSAGE"
git tag -a "$RELEASE_TAG" -m "kvbench $RELEASE_TAG"
SWITCHED=0
echo "created branch $RELEASE_BRANCH at $(git rev-parse --short HEAD) and tag $RELEASE_TAG"

# --- 8. Return to main and restore the untracked paper files -------------------------
# Leaving the release branch removes files that are tracked only there (paper/ files,
# LICENSE, NOTICE) and the directories they leave empty. Restore each paper entry from the
# staged copy, which kept the original owner, mode, and timestamps; paper/ itself is not copied.
git switch -q "$SOURCE_REF"
for entry in "${PAPER_FILES[@]}"; do
    cp -a --remove-destination -- "$STAGE/$entry" "$(dirname -- "$entry")/"
done
stat -c '%u:%g %a %n' -- . paper "${PAPER_FILES[@]}" > "$STAGE/paper_modes_after.txt"
if ! diff -u "$STAGE/paper_modes_before.txt" "$STAGE/paper_modes_after.txt"; then
    echo "warning: owner or mode above differs from before the run; the release commit is unaffected" >&2
fi
echo "back on $SOURCE_REF; nothing was pushed"
