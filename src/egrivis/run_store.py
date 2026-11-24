"""Writes the run bundle: manifest, exit log, occupants, trajectory and plot."""

import csv
import json
import platform
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import jupedsim as jps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from egrivis.runner import RunResult, run
from egrivis.scenario import PrivateScenario

OUTPUT_SCHEMA_VERSION = 1
FRAME_INTERVAL = 0.1  # seconds between stored trajectory frames


def run_and_save(
    scenario: PrivateScenario, seed: int, policy: str = "B1", root: Path = Path("runs")
) -> Path:
    """Run and save. A technical failure is recorded in the manifest and re-raised."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = Path(root) / f"{scenario.id}_{policy}_s{seed}_{stamp}"
    folder.mkdir(parents=True)
    trajectory = folder / "trajectory.sqlite"
    manifest = _base_manifest(scenario, seed, policy)
    try:
        writer = jps.SqliteTrajectoryWriter(
            output_file=trajectory,
            every_nth_frame=max(1, round(FRAME_INTERVAL / scenario.dt)),
        )
        try:
            result = run(scenario, seed, policy, writer)
        finally:
            writer.close()
        _write_tables(folder, result)
        _plot(scenario, trajectory, folder / "paths.png")
    except Exception as error:
        _write_manifest(folder, manifest | {"status": "failed", "error": repr(error)})
        raise
    counts = Counter(c.exit for c in result.crossings)
    _write_manifest(
        folder,
        manifest
        | {
            "status": result.status,
            "end_time": result.end_time,
            "evacuated": len(result.crossings),
            "exit_counts": {e: counts[e] for e in scenario.layout.exits},
            "guidance": result.guidance,
        },
    )
    return folder


def _base_manifest(scenario: PrivateScenario, seed: int, policy: str) -> dict:
    packages = ("egrivis", "jupedsim", "numpy", "shapely")
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "scenario": scenario.id,
        "scenario_sha256": scenario.digest,
        "layout": scenario.layout.id,
        "layout_version": scenario.layout.version,
        "policy": policy,
        "seed": seed,
        "random_streams": ["placement", "desired_speed"],
        "horizon": scenario.horizon,
        "dt": scenario.dt,
        "initial_occupants": scenario.occupant_count,
        "versions": {
            "python": platform.python_version(),
            **{p: version(p) for p in packages},
        },
    }


def _write_manifest(folder: Path, manifest: dict) -> None:
    manifest = manifest | {"files": sorted(p.name for p in folder.iterdir())}
    text = json.dumps(manifest, indent=2)
    (folder / "manifest.json").write_text(text + "\n", encoding="utf-8")


def _write_tables(folder: Path, result: RunResult) -> None:
    with open(folder / "exits.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["time", "occupant", "exit"])
        out.writerows([f"{c.time:.3f}", c.occupant, c.exit] for c in result.crossings)
    with open(folder / "occupants.csv", "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        # agent_id is the ID used in trajectory.sqlite
        out.writerow(["occupant", "agent_id", "x", "y", "desired_speed"])
        out.writerows(
            [o.id, o.agent_id, *o.position, o.desired_speed] for o in result.occupants
        )


def _plot(scenario: PrivateScenario, trajectory: Path, target: Path) -> None:
    layout = scenario.layout
    figure = Figure(figsize=(10, 7))
    FigureCanvasAgg(figure)
    axes = figure.add_subplot()
    for ring in (layout.walkable.exterior, *layout.walkable.interiors):
        axes.plot(*ring.xy, color="black", linewidth=1.5)

    paths = defaultdict(list)
    with closing(sqlite3.connect(trajectory)) as db:
        for agent, x, y in db.execute(
            "SELECT id, pos_x, pos_y FROM trajectory_data ORDER BY frame"
        ):
            paths[agent].append((x, y))
    for points in paths.values():
        axes.plot(*zip(*points), color="tab:blue", linewidth=0.6, alpha=0.5)
        axes.plot(*points[0], ".", color="dimgray", markersize=4)

    for node in layout.nodes.values():
        axes.plot(*node.zone.exterior.xy, color="tab:orange", linestyle="--")
        axes.annotate(node.id, node.position, ha="center", color="tab:orange")
    for exit_ in layout.exits.values():
        axes.fill(*exit_.polygon.exterior.xy, color="tab:green", alpha=0.4)
        axes.plot(*exit_.zone.exterior.xy, color="tab:green", linestyle="--")
        axes.annotate(exit_.id, exit_.position, ha="center", color="tab:green")
    axes.set_aspect("equal")
    axes.set(xlabel="x (m)", ylabel="y (m)", title=f"{scenario.id}: paths")
    figure.savefig(target, dpi=120, bbox_inches="tight")
