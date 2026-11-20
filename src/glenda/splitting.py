"""Build connected groups and verify split isolation; no images are modified."""
from __future__ import annotations

from collections import defaultdict


def connected_groups(rows: list[dict], candidate_pairs: list[tuple[str, str]] | None = None) -> list[str]:
    """Join shared case, video and exact-image identities transitively.

    Missing case identifiers remain unknown, not evidence of different patients.
    """
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    seen = {}
    for i, row in enumerate(rows):
        for field in ("case_id", "video_id", "sha256"):
            value = row.get(field)
            if value:
                key = (field, value)
                if key in seen:
                    union(i, seen[key])
                else:
                    seen[key] = i
    paths = {row["path"]: i for i, row in enumerate(rows)}
    if len(paths) != len(rows):
        raise ValueError("Duplicate paths in inventory")
    for left, right in candidate_pairs or []:
        if left not in paths or right not in paths:
            raise ValueError("Candidate pair refers to a missing inventory path")
        union(paths[left], paths[right])
    # Stable component identifiers independent of union traversal order.
    members = defaultdict(list)
    for i, row in enumerate(rows):
        members[find(i)].append(row["path"])
    return [min(members[find(i)]) for i in range(len(rows))]


def assert_disjoint(rows: list[dict], assignments: list[str], candidate_pairs: list[tuple[str, str]] | None = None) -> None:
    if len(rows) != len(assignments) or not rows:
        raise ValueError("Each nonempty inventory row needs one assignment")
    seen = {}
    for group, split in zip(connected_groups(rows, candidate_pairs), assignments):
        if not split:
            raise ValueError("Empty split assignment")
        if group in seen and seen[group] != split:
            raise ValueError(f"Connected source or duplicate crosses splits: {group}")
        seen[group] = split
