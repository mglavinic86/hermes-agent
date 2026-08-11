# Immutable Change Review Snapshot Runbook

This is a non-live provisioning plan for copying `immutable-change-reviews`
into an isolated `turpi_review` profile. It does not authorize editing an
active profile. The companion manifest template is
[`kanban/immutable-change-reviews.snapshot.yaml.template`](kanban/immutable-change-reviews.snapshot.yaml.template).

## Safety boundary

- Run only against an owner-approved, offline source directory and an empty
  staging directory.
- Resolve both directories before copying. Reject equal, nested, or
  machine-global/live profile destinations.
- Reject every symlink (including symlinked parent components), socket, device,
  FIFO, hard-linked file (`st_nlink != 1`), and path that escapes the source.
- Copy every regular file. There is no exclude list and no live inheritance.
- Do not copy `.env`, credentials, tokens, caches, Git metadata, or absolute
  paths into the manifest. A source containing any of those must be rejected,
  not silently filtered.

## Exact tree digest (`turpi-skill-snapshot-v1`)

The generated manifest must replace `REQUIRED_64_LOWERCASE_HEX` with the
lowercase SHA-256 of this exact byte stream:

1. ASCII bytes `turpi-skill-snapshot-v1\0`.
2. Enumerate every regular file by its UTF-8, POSIX-style path relative to the
   snapshot root, sorted by the raw UTF-8 path bytes. Reject duplicate or
   non-UTF-8 paths.
3. For each file append, without separators:
   - relative-path byte length as an unsigned 64-bit big-endian integer;
   - relative-path UTF-8 bytes;
   - raw file-content byte length as an unsigned 64-bit big-endian integer;
   - raw file-content bytes.

Do not normalize line endings, permissions, timestamps, ownership, or Unicode;
they are not part of the digest. The raw bytes and relative names are. The
manifest itself and any digest receipt are outside the snapshot root.

## Staged procedure

1. Copy the manifest template beside an empty staging directory.
2. Set only relative logical names in the manifest; keep actual machine paths
   in the operator's ephemeral command invocation, never in the artifact.
3. Validate the source contains exactly one root `SKILL.md` whose frontmatter
   name is `immutable-change-reviews`.
4. Apply the safety checks above, compute the framed digest twice from separate
   directory walks, and require identical results.
5. Copy the tree into `skills/immutable-change-reviews` under staging using
   regular-file creation (no links), then recompute the digest from staging.
6. Write the exact digest into `expected_tree_sha256`; fail if it is absent,
   malformed, or differs from either source/staging digest.
7. Review the staged tree and manifest. Promotion into any live profile is a
   separate, explicit owner operation outside this runbook and this change.

## Pinned source snapshot

The companion concrete manifest
[`kanban/immutable-change-reviews.snapshot.yaml`](kanban/immutable-change-reviews.snapshot.yaml)
pins the owner-validated six-file source tree to exact
`turpi-skill-snapshot-v1` SHA-256
`1e74b219dbf5377886fde11fcc873c673d3c01178a007ed9667a0942f8f10ec1`.
The `.template` remains the reusable schema for future snapshots. Provisioning
must reject any source or staging tree that does not reproduce the concrete
manifest's file inventory, per-file hashes, and tree digest.

The manifest is only a content contract. No snapshot has been copied into a
live profile, and live promotion remains a separate owner-approved operation.
