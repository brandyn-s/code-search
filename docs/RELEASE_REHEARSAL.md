# Release rehearsal

The release workflow has three parts that only run for real on `main`:
GitHub artifact attestation, immutable release publication, and PyPI trusted
publishing. Rehearse them with a pre-release before the first real tag so the
first surprise is not on `v0.4.0`.

Pre-release versions follow PEP 440: `X.Y.ZrcN`. The workflow tags them
`vX.Y.ZrcN`, marks the GitHub release as a pre-release, does not mark it
"latest", and publishes to PyPI as a pre-release. `uvx code-search-mcp` and
`pip install code-search-mcp` ignore pre-releases unless the version is
pinned, so a rehearsal never reaches users by accident.

## Prerequisites

- The repository is public (attestations need that outside GitHub Enterprise).
- PyPI project `code-search-mcp` exists with a pending trusted publisher:
  owner `brandyn-s`, repository `code-search`, workflow `release.yml`,
  environment `pypi`. The `pypi` environment exists in the repository settings.
- `Merge CI` has passed on the commit you will release.

## Steps

1. Bump `version` in `pyproject.toml` to `0.4.0rc1`, add nothing to the
   CHANGELOG (rehearsals are not releases), and merge to `main`.
2. Wait for `Merge CI` on that commit, then dispatch `Release` with version
   `0.4.0rc1`.
3. Verify the outputs:

   ```bash
   REPO=brandyn-s/code-search TAG=v0.4.0rc1
   gh release view "$TAG" --repo "$REPO" --json isPrerelease,isDraft,assets \
     --jq '{isPrerelease, isDraft, assets: [.assets[].name]}'
   gh release download "$TAG" --repo "$REPO" --dir /tmp/rc
   gh attestation verify /tmp/rc/code_search_mcp-0.4.0rc1-py3-none-any.whl \
     --bundle /tmp/rc/code_search_mcp-0.4.0rc1-provenance.jsonl \
     --repo "$REPO" --signer-workflow "$REPO/.github/workflows/release.yml"
   gh release verify "$TAG" --repo "$REPO"
   uvx code-search-mcp==0.4.0rc1 --help
   uvx code-search-mcp==0.4.0rc1 doctor --no-network
   uvx code-search-mcp --help   # must NOT resolve to the rc
   ```

   Expect `isPrerelease: true`, three assets, both verifications passing, and
   the last command either failing to resolve (no final release yet) or
   resolving to the previous final release.

4. Clean up so the rehearsal leaves no trace users could hit:

   ```bash
   # PyPI: yank the pre-release (keeps the record, hides it from resolvers)
   # via https://pypi.org/manage/project/code-search-mcp/release/0.4.0rc1/
   gh release delete "$TAG" --repo "$REPO" --yes --cleanup-tag
   ```

5. Set `version` back to `0.4.0`, move the `Unreleased` CHANGELOG section
   under `0.4.0`, merge, and dispatch `Release` with `0.4.0`.

## What a failure tells you

| Stage that failed | Likely cause |
|---|---|
| preflight `test "$VERSION" = "$PROJECT_VERSION"` | pyproject version not bumped |
| preflight "No successful Unit Tests push run" | `Merge CI` has not finished on this commit |
| attest | repository is private, or `id-token: write` missing |
| publish (GitHub) | tag already exists from an earlier attempt; delete it with `--cleanup-tag` |
| publish-pypi | trusted publisher not configured for this workflow/environment, or the `pypi` environment is missing |
