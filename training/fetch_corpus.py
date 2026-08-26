#!/usr/bin/env python3
"""Fetch the *rich combined wound corpus* used across all three training tasks.

This CLI pulls together every **openly, non-interactively fetchable** wound
dataset the dataset audit verified, into one on-disk data root, so that
``training/data_prep.py`` (driven by ``training/configs/datasets.yaml``) can
aggregate them into leakage-free manifests for:

* **segmentation** — real binary/instance wound masks;
* **MedSAM mask-factory** — staged / boxed-but-maskless sets that become mask
  targets once run through MedSAM box/point prompts;
* **teacher-student grading** — pressure-injury stage labels and orthogonal
  tissue-type / depth labels.

It fetches ONLY sources whose access the audit confirmed as programmatic:

* ``fuseg``    — reuses the repo's ``scripts/fetch_fuseg.py`` (raw GitHub);
* ``git``      — ``git clone`` of a public repository;
* ``mendeley`` — Mendeley Data public files API (CC-BY, no login);
* ``kaggle``   — ``kaggle`` CLI (needs a free API token in the environment);
* ``roboflow`` — Roboflow SDK export (needs a free ``ROBOFLOW_API_KEY``).

Gated or registration-only sources (DFUC, WoundsDB, Medetec, and the two
tissue sets whose hosts the audit could not verify) are **never** downloaded:
for those this tool prints the exact manual step and the local path where the
prepared data must land, then moves on.

Every URL here is copied from the dataset audit; none are invented. Nothing is
fetched unless you pass ``--data-root`` without ``--list``/``--dry-run``.

Examples
--------
List the whole corpus with fetch method and task tags::

    python training/fetch_corpus.py --list

Dry-run the plan (print commands, download nothing)::

    python training/fetch_corpus.py --data-root data/corpus --dry-run

Fetch only the open segmentation sources::

    python training/fetch_corpus.py --data-root data/corpus \
        --source azh_fuseg --source co2wounds_v2 --source lowerlimb_feet
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FUSEG_SCRIPT = REPO_ROOT / "scripts" / "fetch_fuseg.py"

# Task tags used across the corpus (kept in sync with datasets.yaml comments).
TASK_SEG = "segmentation"
TASK_MASK_FACTORY = "medsam-mask-factory"
TASK_GRADING = "teacher-student-grading"


@dataclass(frozen=True, slots=True)
class CorpusSource:
    """One dataset in the combined corpus and how to obtain it.

    Attributes
    ----------
    name:
        Stable id, also the sub-directory created under ``--data-root`` and the
        ``name`` to reference from ``training/configs/datasets.yaml``.
    method:
        Fetch mechanism: ``"fuseg"``, ``"git"``, ``"mendeley"``, ``"kaggle"``,
        ``"roboflow"`` (all programmatic) or ``"manual"`` (gated / unverified
        host — instructions only, never downloaded).
    tasks:
        Which training tasks the source feeds.
    url:
        The audit-provided access URL (landing page or clone URL).
    body_part:
        Coarse anatomical coverage, for the manifest/config.
    detail:
        Method-specific payload: a git clone URL, a ``"{id}:{version}"`` for
        Mendeley, a ``"owner/dataset"`` slug for Kaggle, a
        ``"workspace/project"`` slug for Roboflow, or "" for fuseg/manual.
    subdir:
        Sub-path (relative to a source's root dir) that actually holds the data
        once fetched, when the download nests it (e.g. a mono-repo folder).
    manual_step:
        For gated/manual sources, the exact human action required.
    note:
        Short freeform note (license flags, guideline caveats).
    """

    name: str
    method: str
    tasks: tuple[str, ...]
    url: str
    body_part: str
    detail: str = ""
    subdir: str = ""
    manual_step: str = ""
    note: str = ""

    @property
    def fetchable(self) -> bool:
        """Whether this source can be pulled without human interaction."""
        return self.method != "manual"

    def dest(self, data_root: Path) -> Path:
        """Absolute directory this source is fetched into."""
        return data_root / self.name


# --------------------------------------------------------------------------- #
# The corpus registry — every entry's URL is taken verbatim from the audit.
# --------------------------------------------------------------------------- #
REGISTRY: tuple[CorpusSource, ...] = (
    # ---- Open, real masks: segmentation core -------------------------------
    CorpusSource(
        name="azh_fuseg",
        method="fuseg",
        tasks=(TASK_SEG,),
        url="https://github.com/uwm-bigdata/wound-segmentation",
        body_part="foot",
        note="AZH + FUSeg foot-ulcer binary masks; research-only, no grade.",
    ),
    CorpusSource(
        name="lowerlimb_feet",
        method="mendeley",
        tasks=(TASK_SEG, TASK_GRADING, TASK_MASK_FACTORY),
        url="https://data.mendeley.com/datasets/hsj38fwnvr/3",
        body_part="lower_limb",
        detail="hsj38fwnvr:3",
        note="CC-BY-4.0; 2,686 masked + 8 wound-type labels (no NPIAP stage).",
    ),
    CorpusSource(
        name="co2wounds_v2",
        method="mendeley",
        tasks=(TASK_SEG,),
        url="https://data.mendeley.com/datasets/s2w7rjwz49/2",
        body_part="mixed",
        detail="s2w7rjwz49:2",
        note="764 leprosy chronic-wound imgs, COCO polygons + binary masks; "
        "code at github.com/simatec-uis/CO2Wounds-V2.",
    ),
    CorpusSource(
        name="wsnet",
        method="git",
        tasks=(TASK_SEG,),
        url="https://github.com/subbareddy248/WSNET",
        body_part="mixed",
        detail="https://github.com/subbareddy248/WSNET",
        note="Multi-type wound masks (8 wound types), mixed body parts; research.",
    ),
    # ---- Open, tissue-type / depth: grading teachers -----------------------
    CorpusSource(
        name="dfutissue",
        method="git",
        tasks=(TASK_GRADING,),
        url="https://github.com/uwm-bigdata/DFUTissueSegNet",
        body_part="foot",
        detail="https://github.com/uwm-bigdata/DFUTissueSegNet",
        subdir="DFUTissue",
        note="110 imgs, 8 tissue classes incl. tendon/bone (depth); license "
        "unstated (research/AZH) — flag before non-research use.",
    ),
    CorpusSource(
        name="complexwounddb",
        method="git",
        tasks=(TASK_GRADING,),
        url="https://github.com/recogna-lab/datasets/tree/master/ComplexWoundDB",
        body_part="mixed",
        detail="https://github.com/recogna-lab/datasets",
        subdir="ComplexWoundDB",
        note="27 in-the-wild imgs, 5 tissue classes; license 'free available' "
        "(unspecified) — pilot only.",
    ),
    # ---- Open, stage labels: grading teachers ------------------------------
    CorpusSource(
        name="piid",
        method="git",
        tasks=(TASK_GRADING,),
        url="https://github.com/FU-MedicalAI/PIID",
        body_part="mixed",
        detail="https://github.com/FU-MedicalAI/PIID",
        note="~1,091 whole-image stage labels (EPUAP I-IV per most sources), "
        "no masks; no explicit license — research risk.",
    ),
    CorpusSource(
        name="roboflow_stages",
        method="roboflow",
        tasks=(TASK_GRADING, TASK_MASK_FACTORY),
        url="https://universe.roboflow.com/stage2-n7xya/pressure-ulcer-sxitf",
        body_part="mixed",
        detail="stage2-n7xya/pressure-ulcer-sxitf",
        note="~2,811 imgs, stage1-4 bounding boxes (guideline NPIAP/EPUAP "
        "unconfirmed); boxes are MedSAM prompts. Needs ROBOFLOW_API_KEY.",
    ),
    CorpusSource(
        name="kaggle_stages",
        method="kaggle",
        tasks=(TASK_GRADING, TASK_MASK_FACTORY),
        url="https://www.kaggle.com/datasets/sinemgokoz/pressure-ulcers-stages",
        body_part="mixed",
        detail="sinemgokoz/pressure-ulcers-stages",
        note="Pressure-ulcer stage classes, multi-site, no masks (guideline "
        "unconfirmed). Needs a Kaggle API token.",
    ),
    # ---- Gated / registration / unverified host: manual only ---------------
    CorpusSource(
        name="dfuc",
        method="manual",
        tasks=(TASK_SEG, TASK_GRADING),
        url="https://dfu-challenge.github.io",
        body_part="foot",
        manual_step="Apply and accept the DFUC data-use agreement (email / "
        "Grand Challenge registration), then arrange the segmentation editions "
        "as images/ + masks/ (adapter: generic). DFUC 2021 patch labels feed "
        "the grading head, not segmentation.",
        note="GATED (DUA, non-commercial).",
    ),
    CorpusSource(
        name="woundsdb",
        method="manual",
        tasks=(TASK_SEG,),
        url="https://chronicwounddatabase.eu",
        body_part="mixed",
        manual_step="Register at chronicwounddatabase.eu via a browser (blocks "
        "scrapers). Export RGB photos + expert outline masks into images/ + "
        "masks/. Recover per-patient visit grouping if you can (weakly "
        "longitudinal).",
        note="Registration-only (browser).",
    ),
    CorpusSource(
        name="medetec",
        method="manual",
        tasks=(TASK_MASK_FACTORY,),
        url="https://www.medetec.co.uk",
        body_part="mixed",
        manual_step="Download the aetiology-labelled stock photos from "
        "medetec.co.uk (no masks ship with them). Use as an unlabeled / "
        "MedSAM-pseudo-labelled pool; import with require_masks: false.",
        note="Open free reuse, but maskless and no bulk endpoint.",
    ),
    CorpusSource(
        name="wounds_307",
        method="manual",
        tasks=(TASK_GRADING,),
        url="https://www.nature.com/articles/s41598-025-06703-5",
        body_part="mixed",
        manual_step="Resolve the data-availability statement of Sci. Reports "
        "s41598-025-06703-5 (host UNVERIFIED by the audit — likely author "
        "request). Place granulation/slough/eschar tissue masks as generic.",
        note="Host UNVERIFIED — do not assume a direct link.",
    ),
    CorpusSource(
        name="woundtissue_147",
        method="manual",
        tasks=(TASK_GRADING,),
        url="https://arxiv.org/abs/2502.10652",
        body_part="mixed",
        manual_step="Contact the authors of arXiv:2502.10652 for the 147-image "
        "6-tissue set (incl. bone/tendon depth). Host UNVERIFIED by the audit.",
        note="Host UNVERIFIED — request/DUA until confirmed.",
    ),
)

REGISTRY_BY_NAME: dict[str, CorpusSource] = {s.name: s for s in REGISTRY}

_MENDELEY_API = "https://data.mendeley.com/public-api/datasets/{id}/files?folder_id=&version={ver}"


@dataclass(slots=True)
class FetchResult:
    """Outcome of attempting one source."""

    name: str
    status: str  # "fetched", "skipped-manual", "planned", "error"
    detail: str = ""
    manual_paths: list[str] = field(default_factory=list)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    """Run ``cmd``, or print it under ``--dry-run``."""
    printable = " ".join(cmd)
    if dry_run:
        print(f"    [dry-run] {printable}")
        return
    print(f"    $ {printable}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Per-method fetchers
# --------------------------------------------------------------------------- #
def fetch_fuseg(
    source: CorpusSource, dest: Path, *, dry_run: bool, fuseg_limit: int
) -> FetchResult:
    """Fetch AZH + FUSeg foot-ulcer masks via ``scripts/fetch_fuseg.py``."""
    for split in ("train", "validation"):
        _run(
            [
                sys.executable,
                str(FUSEG_SCRIPT),
                "--out",
                str(dest),
                "--split",
                split,
                "--limit",
                str(fuseg_limit),
            ],
            dry_run=dry_run,
        )
    return FetchResult(source.name, "planned" if dry_run else "fetched", str(dest))


def fetch_git(source: CorpusSource, dest: Path, *, dry_run: bool, depth: int) -> FetchResult:
    """Shallow-clone a public git repository."""
    if dest.exists() and not dry_run:
        print(f"    exists, skipping clone: {dest}")
        return FetchResult(source.name, "fetched", str(dest))
    _run(
        ["git", "clone", "--depth", str(depth), source.detail, str(dest)],
        dry_run=dry_run,
    )
    return FetchResult(source.name, "planned" if dry_run else "fetched", str(dest))


def _mendeley_files(dataset_id: str, version: str) -> list[dict[str, object]]:
    """List a Mendeley dataset's files via the public (no-login) files API."""
    url = _MENDELEY_API.format(id=dataset_id, ver=version)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        kind = type(payload).__name__
        raise ValueError(f"unexpected Mendeley API payload for {dataset_id}: {kind}")
    return payload


def _mendeley_download_url(entry: dict[str, object]) -> str | None:
    """Extract a direct download URL from one Mendeley file entry."""
    direct = entry.get("download_url")
    if isinstance(direct, str) and direct:
        return direct
    details = entry.get("content_details")
    if isinstance(details, dict):
        nested = details.get("download_url")
        if isinstance(nested, str) and nested:
            return nested
    return None


def fetch_mendeley(source: CorpusSource, dest: Path, *, dry_run: bool) -> FetchResult:
    """Fetch a CC-BY Mendeley dataset via its public files API."""
    dataset_id, version = source.detail.split(":", 1)
    if dry_run:
        print(f"    [dry-run] GET {_MENDELEY_API.format(id=dataset_id, ver=version)}")
        print(f"    [dry-run] download each listed file into {dest}")
        return FetchResult(source.name, "planned", str(dest))
    try:
        files = _mendeley_files(dataset_id, version)
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        hint = (
            f"Mendeley public API unreachable/changed for {dataset_id} v{version} "
            f"({exc}). Download the zip manually from {source.url} into {dest}."
        )
        print(f"    ! {hint}")
        return FetchResult(source.name, "error", hint, manual_paths=[str(dest)])
    dest.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for entry in files:
        download_url = _mendeley_download_url(entry)
        filename = entry.get("filename")
        if download_url is None or not isinstance(filename, str):
            continue
        target = dest / filename
        print(f"    $ download {filename}")
        with urllib.request.urlopen(download_url, timeout=120) as response:
            target.write_bytes(response.read())
        fetched += 1
    return FetchResult(source.name, "fetched", f"{fetched} files -> {dest}")


def fetch_kaggle(source: CorpusSource, dest: Path, *, dry_run: bool) -> FetchResult:
    """Fetch a Kaggle dataset via the ``kaggle`` CLI (needs an API token)."""
    if shutil.which("kaggle") is None and not dry_run:
        hint = (
            "kaggle CLI not found. Install it and place a token at "
            f"~/.kaggle/kaggle.json, then: kaggle datasets download -d "
            f"{source.detail} -p {dest} --unzip"
        )
        print(f"    ! {hint}")
        return FetchResult(source.name, "error", hint, manual_paths=[str(dest)])
    _run(
        ["kaggle", "datasets", "download", "-d", source.detail, "-p", str(dest), "--unzip"],
        dry_run=dry_run,
    )
    return FetchResult(source.name, "planned" if dry_run else "fetched", str(dest))


def fetch_roboflow(
    source: CorpusSource, dest: Path, *, dry_run: bool, version: int, fmt: str
) -> FetchResult:
    """Export a Roboflow Universe project via the SDK (needs ROBOFLOW_API_KEY)."""
    import os

    workspace, project = source.detail.split("/", 1)
    if dry_run:
        print(
            f"    [dry-run] roboflow: workspace={workspace} project={project} "
            f"version={version} format={fmt} -> {dest}"
        )
        return FetchResult(source.name, "planned", str(dest))
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        hint = (
            "ROBOFLOW_API_KEY not set. Get a free key at roboflow.com and export "
            f"ROBOFLOW_API_KEY, then re-run to export {source.detail} into {dest}."
        )
        print(f"    ! {hint}")
        return FetchResult(source.name, "error", hint, manual_paths=[str(dest)])
    try:
        from roboflow import Roboflow
    except ImportError:
        hint = "roboflow package not installed: pip install roboflow, then re-run."
        print(f"    ! {hint}")
        return FetchResult(source.name, "error", hint, manual_paths=[str(dest)])
    rf = Roboflow(api_key=api_key)
    handle = rf.workspace(workspace).project(project).version(version)
    handle.download(fmt, location=str(dest))
    return FetchResult(source.name, "fetched", f"{fmt} v{version} -> {dest}")


def print_manual(source: CorpusSource, dest: Path) -> FetchResult:
    """Print the manual step for a gated/unverified source (no download)."""
    print(f"  {source.name}: MANUAL — {source.note}")
    print(f"    step: {source.manual_step}")
    print(f"    place prepared data at: {dest}")
    return FetchResult(source.name, "skipped-manual", source.manual_step, manual_paths=[str(dest)])


# --------------------------------------------------------------------------- #
# Dispatch + listing
# --------------------------------------------------------------------------- #
def _dispatch(source: CorpusSource, args: argparse.Namespace) -> FetchResult:
    """Fetch one source according to its method."""
    dest = source.dest(args.data_root)
    if source.method == "manual":
        return print_manual(source, dest)

    print(f"  {source.name}: {source.method} -> {dest}")
    if source.method == "fuseg":
        return fetch_fuseg(source, dest, dry_run=args.dry_run, fuseg_limit=args.fuseg_limit)
    if source.method == "git":
        return fetch_git(source, dest, dry_run=args.dry_run, depth=args.git_depth)
    if source.method == "mendeley":
        return fetch_mendeley(source, dest, dry_run=args.dry_run)
    if source.method == "kaggle":
        return fetch_kaggle(source, dest, dry_run=args.dry_run)
    if source.method == "roboflow":
        return fetch_roboflow(
            source,
            dest,
            dry_run=args.dry_run,
            version=args.roboflow_version,
            fmt=args.roboflow_format,
        )
    raise ValueError(f"unknown fetch method {source.method!r} for {source.name}")


def print_listing() -> None:
    """Print every source with its fetch method and task tags."""
    print("Rich combined wound corpus — sources")
    print("=" * 78)
    for source in REGISTRY:
        access = "OPEN " if source.fetchable else "GATED"
        tags = ",".join(t.split("-")[0] for t in source.tasks)
        print(f"[{access}] {source.name:16s} method={source.method:9s} tasks={tags}")
        print(f"          url:  {source.url}")
        if source.subdir:
            print(f"          data: <data-root>/{source.name}/{source.subdir}")
        if source.note:
            print(f"          note: {source.note}")
    print("=" * 78)
    fetchable = sum(1 for s in REGISTRY if s.fetchable)
    print(f"{fetchable} programmatic, {len(REGISTRY) - fetchable} manual, {len(REGISTRY)} total.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch the open, programmatically available wound corpus.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory to fetch sources into (required unless --list).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every source with its fetch method and task tags, then exit.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="NAME",
        help="Fetch only this source (repeatable). Default: all open sources "
        "plus manual-step notices.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned fetch commands without downloading anything.",
    )
    parser.add_argument(
        "--fuseg-limit",
        type=int,
        default=1200,
        help="Max FUSeg image/mask pairs per split (passed to fetch_fuseg.py).",
    )
    parser.add_argument("--git-depth", type=int, default=1, help="git clone --depth.")
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=1,
        help="Roboflow project version to export.",
    )
    parser.add_argument(
        "--roboflow-format",
        default="coco",
        help="Roboflow export format (e.g. coco, coco-segmentation, voc).",
    )
    return parser.parse_args(argv)


def _selected_sources(names: list[str] | None) -> list[CorpusSource]:
    """Resolve ``--source`` selection to registry entries."""
    if not names:
        return list(REGISTRY)
    resolved: list[CorpusSource] = []
    for name in names:
        source = REGISTRY_BY_NAME.get(name)
        if source is None:
            raise SystemExit(
                f"unknown source {name!r}; choose from: {', '.join(REGISTRY_BY_NAME)}"
            )
        resolved.append(source)
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    if args.list:
        print_listing()
        return 0

    if args.data_root is None:
        raise SystemExit("--data-root is required unless --list is given")

    args.data_root = args.data_root.expanduser().resolve()
    args.data_root.mkdir(parents=True, exist_ok=True)
    print(f"data root: {args.data_root}" + ("  (dry-run)" if args.dry_run else ""))

    results = [_dispatch(source, args) for source in _selected_sources(args.source)]

    print("\nsummary")
    print("-" * 40)
    for result in results:
        print(f"  {result.name:16s} {result.status:14s} {result.detail}")
    manual = [r for r in results if r.status in ("skipped-manual", "error")]
    if manual:
        print(f"\n{len(manual)} source(s) need a manual step — see the lines above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
