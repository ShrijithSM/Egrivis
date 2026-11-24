from egrivis.routing import static_shortest_path
from egrivis.scenario import parse_scenario


def test_b1_sends_everyone_to_the_nearer_exit(raw):
    guidance = static_shortest_path(parse_scenario(raw("two_exit")).layout)
    assert guidance == {
        "door": "door_to_junction",
        "junction": "junction_to_a",
        "exit_a": None,
        "exit_b": None,
    }


def test_ties_go_to_the_lowest_edge_id(raw):
    data = raw("two_exit")
    # Move the door, junction and their zones to the middle, 15 m from each end.
    outer = data["layout"]["walkable"]["outer"]
    for point in outer:
        point[0] += {10: 4, 12: 4}.get(point[0], 0)
    for node in data["layout"]["nodes"]:
        node["position"][0] += 4
        for point in node["zone"]:
            point[0] += 4
    for _ in range(2):
        layout = parse_scenario(data).layout
        assert static_shortest_path(layout)["junction"] == "junction_to_a"
        data["layout"]["edges"].reverse()


def test_unreachable_node_gets_no_route(raw):
    data = raw("two_exit")
    data["layout"]["edges"] = [
        e for e in data["layout"]["edges"] if e["source"] != "door"
    ]
    data["spawns"][0]["target"] = "junction"
    guidance = static_shortest_path(parse_scenario(data).layout)
    assert guidance["door"] is None
    assert guidance["junction"] == "junction_to_a"


def test_hidden_events_do_not_change_b1(raw):
    open_, closing = (
        static_shortest_path(parse_scenario(raw(n)).layout)
        for n in ("two_exit", "two_exit_closure")
    )
    assert open_ == closing
