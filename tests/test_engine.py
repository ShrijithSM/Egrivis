"""Checks of the JuPedSim behaviour that Egrivis relies on."""

import jupedsim as jps
import pytest
import shapely

CORRIDOR = [(0, 0), (20, 0), (20, 3), (0, 3)]
EXIT = [(19, 0), (20, 0), (20, 3), (19, 3)]


def make_sim():
    return jps.Simulation(
        model=jps.CollisionFreeSpeedModelV2(), geometry=CORRIDOR, dt=0.01
    )


def add_journey(sim, stage):
    return sim.add_journey(jps.JourneyDescription([stage]))


def add_agent(sim, journey, stage, pos=(2.0, 1.5), speed=1.2):
    return sim.add_agent(
        jps.CollisionFreeSpeedModelV2AgentParameters(
            position=pos, desired_speed=speed, journey_id=journey, stage_id=stage
        )
    )


def step(sim, seconds):
    removed = []
    for _ in range(round(seconds / 0.01)):
        sim.iterate()
        removed += sim.removed_agents()
    return removed


def test_corridor_travel_time():
    sim = make_sim()
    stage = sim.add_exit_stage(EXIT)
    add_agent(sim, add_journey(sim, stage), stage)
    while sim.agent_count():
        sim.iterate()
        assert sim.elapsed_time() < 60
    assert sim.elapsed_time() == pytest.approx(17 / 1.2, rel=0.05)


def test_journey_switch_mid_run():
    sim = make_sim()
    right = sim.add_waypoint_stage((18, 1.5), 0.3)
    left = sim.add_waypoint_stage((1, 1.5), 0.3)
    j_right, j_left = add_journey(sim, right), add_journey(sim, left)
    agent = add_agent(sim, j_right, right)
    step(sim, 5)
    x = sim.agent(agent).position[0]
    assert x > 5
    sim.switch_agent_journey(agent, j_left, left)
    step(sim, 3)
    assert sim.agent(agent).position[0] < x - 2


def test_exit_ignores_agents_on_other_journeys():
    sim = make_sim()
    sim.add_exit_stage(EXIT)
    inside = sim.add_waypoint_stage((19.5, 1.5), 0.2)
    agent = add_agent(sim, add_journey(sim, inside), inside)
    assert step(sim, 30) == []
    assert sim.agent(agent).position[0] > 19


def test_released_agent_can_be_recalled():
    sim = make_sim()
    exit_stage = sim.add_exit_stage(EXIT)
    gate = sim.add_waypoint_stage((16, 1.5), 0.3)
    j_exit, j_gate = add_journey(sim, exit_stage), add_journey(sim, gate)
    agent = add_agent(sim, j_exit, exit_stage)
    while sim.agent(agent).position[0] < 18.5:
        sim.iterate()
    sim.switch_agent_journey(agent, j_gate, gate)
    assert step(sim, 20) == []
    assert sim.agent(agent).position[0] < 17


def crowd_run():
    sim = make_sim()
    stage = sim.add_exit_stage(EXIT)
    journey = add_journey(sim, stage)
    pts = jps.distribute_by_number(
        polygon=shapely.box(0, 0, 8, 3),
        number_of_agents=20, distance_to_agents=0.5, distance_to_polygon=0.3, seed=1,
    )
    for p in pts:
        add_agent(sim, journey, stage, p)
    log = []
    while sim.agent_count():
        sim.iterate()
        log += [(sim.iteration_count(), a) for a in sim.removed_agents()]
        assert sim.elapsed_time() < 100
    base = min(a for _, a in log)
    return [(i, a - base) for i, a in log]


def test_crowd_run_repeats():
    assert crowd_run() == crowd_run()
