import csv
import json
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest

from egrivis import run_store
from egrivis.run_store import run_and_save
from egrivis.runner import run
from egrivis.scenario import parse_scenario
from egrivis.simulator import Simulator


def log(result):
    return [(c.time, c.occupant, c.exit) for c in result.crossings]


def small(raw, name, count=10, **changes):
    data = raw(name)
    data["spawns"][0]["count"] = count
    data.update(changes)
    return parse_scenario(data)


def event(time, is_open, exit_id="exit_a"):
    return {"time": time, "exit": exit_id, "open": is_open}


def test_single_occupant_walks_at_desired_speed(raw):
    result = run(small(raw, "corridor", count=1), seed=3)
    (occupant,), (crossing,) = result.occupants, result.crossings
    expected = (19 - occupant.position[0]) / occupant.desired_speed
    # 5 % allows for the detour via the centre line and reaction at the waypoint.
    assert crossing.time == pytest.approx(expected, rel=0.05)


def test_everyone_leaves_once(raw):
    result = run(small(raw, "two_exit", count=50), seed=1)
    assert result.status == "finished"
    assert sorted(c.occupant for c in result.crossings) == list(range(50))
    assert {c.exit for c in result.crossings} == {"exit_a"}


def test_same_seed_repeats_and_other_seed_differs(raw):
    scenario = small(raw, "two_exit")
    first = log(run(scenario, seed=1))
    assert first == log(run(scenario, seed=1))
    assert first != log(run(scenario, seed=2))


def test_placement_and_speed_do_not_depend_on_events(raw):
    def traits(scenario):
        return [
            (o.position, o.desired_speed) for o in Simulator(scenario, 1).occupants
        ]

    assert traits(small(raw, "two_exit")) == traits(small(raw, "two_exit_closure"))


def test_nobody_crosses_a_closed_gate_and_flow_resumes(raw):
    scenario = small(
        raw, "two_exit", count=50, exit_events=[event(20, False), event(60, True)]
    )
    result = run(scenario, seed=1)
    times = [c.time for c in result.crossings]
    assert any(t <= 20 for t in times)
    assert not any(20 < t <= 60 for t in times)
    assert result.status == "finished" and len(times) == 50


def test_permanently_closed_exit_runs_to_horizon(raw):
    scenario = small(raw, "two_exit", horizon=60, exit_events=[event(0, False)])
    result = run(scenario, seed=1)
    assert (result.status, result.crossings) == ("horizon", [])
    assert result.end_time == pytest.approx(60)


def test_released_occupants_are_recalled_when_the_gate_closes(raw):
    # Close exactly when the first occupant has been released but not yet crossed.
    free = run(small(raw, "two_exit"), seed=1)
    close_at = round(free.crossings[0].time - 0.3, 2)
    scenario = small(raw, "two_exit", horizon=60, exit_events=[event(close_at, False)])
    assert run(scenario, seed=1).crossings == []


def test_new_guidance_redirects_people_waiting_at_a_closed_gate(raw):
    scenario = small(raw, "two_exit", exit_events=[event(0, False)])
    simulator = Simulator(scenario, seed=1)
    route = {"door": "door_to_junction", "junction": "junction_to_a"}
    simulator.set_guidance(route)
    while simulator.time < 40:
        simulator.step()
    assert not simulator.is_open("exit_a") and simulator.remaining == 10

    simulator.set_guidance(
        {**route, "junction": "junction_to_b", "exit_a": "a_to_junction"}
    )
    while simulator.remaining and simulator.time < 180:
        simulator.step()
    assert [c.exit for c in simulator.crossings] == ["exit_b"] * 10


@pytest.mark.parametrize(
    "guidance, message",
    [
        ({"junction": "door_to_junction"}, "invalid edge"),
        ({"junction": "missing"}, "invalid edge"),
        ({"door": "door_to_junction"}, "stops at junction"),
        ({"exit_a": "a_to_junction", "junction": "junction_to_a"}, "cycle"),
    ],
)
def test_guidance_is_validated(raw, guidance, message):
    simulator = Simulator(small(raw, "two_exit"), seed=1)
    with pytest.raises(ValueError, match=message):
        simulator.set_guidance(guidance)


def test_bundle_matches_the_run(raw, tmp_path):
    folder = run_and_save(small(raw, "two_exit"), seed=1, root=tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    with open(folder / "exits.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(folder / "occupants.csv", newline="") as f:
        agents = {int(r["agent_id"]) for r in csv.DictReader(f)}
    with closing(sqlite3.connect(folder / "trajectory.sqlite")) as db:
        recorded = {a for (a,) in db.execute("SELECT DISTINCT id FROM trajectory_data")}

    assert manifest["status"] == "finished"
    assert manifest["evacuated"] == len(rows) == 10
    assert manifest["exit_counts"] == {"exit_a": 10, "exit_b": 0}
    assert recorded == agents
    assert (folder / "paths.png").stat().st_size > 0


def test_technical_failure_is_recorded(raw, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_store, "run", fail)
    with pytest.raises(RuntimeError):
        run_and_save(small(raw, "two_exit"), seed=1, root=tmp_path)
    (folder,) = tmp_path.iterdir()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "boom" in manifest["error"]


def test_cli_reports_bad_scenarios(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    done = subprocess.run(
        [sys.executable, "-m", "egrivis", "run", str(bad)],
        capture_output=True, text=True,
    )
    assert done.returncode == 2 and done.stderr.startswith("error:")


def test_guidance_only_applies_to_people_heading_for_that_node(raw):
    # Walks through the door and junction zones, but on the way to exit B.
    data = raw("two_exit")
    data["spawns"][0].update(count=10, target="exit_b")
    result = run(parse_scenario(data), seed=1)
    assert [c.exit for c in result.crossings] == ["exit_b"] * 10
