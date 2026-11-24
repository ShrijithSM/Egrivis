import pytest

from egrivis.scenario import ScenarioError, load_scenario, parse_scenario


def set_in(data, path, value):
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value


@pytest.mark.parametrize("name", ["corridor", "two_exit", "two_exit_closure"])
def test_shipped_scenarios_are_valid(raw, name):
    parse_scenario(raw(name))


BOWTIE = [[0, 0], [4, 4], [4, 0], [0, 4]]
OUTSIDE = [[40, 0], [42, 0], [42, 2], [40, 2]]
NODES, EXITS, EDGES = ("layout", "nodes"), ("layout", "exits"), ("layout", "edges")


@pytest.mark.parametrize(
    "path, value, message",
    [
        (("schema_version",), 99, "schema_version"),
        (("layout", "walkable", "outer"), BOWTIE, "invalid polygon"),
        ((*EXITS, 0, "id"), "exit_b", "duplicate IDs"),
        ((*NODES, 0, "zone"), OUTSIDE, "zone leaves walkable"),
        ((*NODES, 0, "position"), [10.0, 5.0], "position not walkable"),
        ((*NODES, 0, "position"), [20, 12], "outside its zone"),
        ((*EXITS, 0, "polygon"), OUTSIDE, "exit polygon leaves walkable"),
        ((*EXITS, 0, "polygon"), [[0, 0], [2, 0], [2, 4], [0, 4]], "overlaps"),
        ((*EDGES, 0, "target"), "nowhere", "unknown endpoint"),
        ((*EDGES, 0, "target"), "door", "source equals target"),
        ((*EDGES, 0, "waypoints"), [[20, 5]], "path leaves walkable"),
        ((*EDGES, 0, "target"), "exit_a", "path leaves walkable"),
        (EDGES, [], "no exit is reachable"),
        (EXITS, [], "at least one exit"),
        (("spawns",), [], "at least one spawn"),
        (("spawns", 0, "polygon"), OUTSIDE, "leaves walkable"),
        (("spawns", 0, "count"), 0, "positive integer"),
        (("spawns", 0, "target"), "nowhere", "unknown target"),
        (("speed", "minimum"), 0, "speed distribution"),
        (("dt",), 0, "must be positive"),
        (("exit_events",), [{"time": 1, "exit": "x", "open": False}], "unknown exit"),
        (
            ("exit_events",),
            [{"time": 181, "exit": "exit_a", "open": False}],
            "outside 0..horizon",
        ),
        (("exit_events",), [{"time": 1, "exit": "exit_a", "open": 0}], "true or false"),
        (("layout", "walkable"), {}, "malformed"),
    ],
)
def test_invalid_scenarios_are_rejected(raw, path, value, message):
    data = raw("two_exit")
    set_in(data, path, value)
    with pytest.raises(ScenarioError, match=message):
        parse_scenario(data)


def test_non_json_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{")
    with pytest.raises(ScenarioError, match="not valid JSON"):
        load_scenario(path)


def test_public_layout_is_independent_of_hidden_events(raw):
    quiet = parse_scenario(raw("two_exit"))
    closing = parse_scenario(raw("two_exit_closure"))
    assert quiet.layout == closing.layout
    assert quiet.digest != closing.digest
