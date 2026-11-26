"""Checks for the procedural scores (libero/libero/envs/procedural_score.py).

The scoring rules run on a stub env. The procedures are checked against the
benchmark's 130 tasks, the BDDL objects, and the saved initial states. Real
scenes check that each new condition holds only when it should: a scripted pick
for `Lifted`, and objects placed by hand for `CarriedOff`.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "libero" / "libero"
SUITES = ["libero_90", "libero_10", "libero_spatial", "libero_object", "libero_goal"]
NO_STAGE_TASKS = {  # one continuous motion (a drawer, a door, a knob, a push, a slide): success only
    ("libero_90", 0), ("libero_90", 6), ("libero_90", 7), ("libero_90", 11), ("libero_90", 20), ("libero_90", 22),
    ("libero_90", 28), ("libero_90", 33), ("libero_90", 35), ("libero_90", 39), ("libero_90", 42), ("libero_90", 44),
    ("libero_goal", 0), ("libero_goal", 5), ("libero_goal", 7),
}
TWO_STAGE_TASKS = {  # every other task has exactly one stage
    ("libero_90", 1), ("libero_90", 3), ("libero_90", 4), ("libero_90", 5), ("libero_90", 8), ("libero_90", 21),
    ("libero_90", 45), ("libero_10", 0), ("libero_10", 1), ("libero_10", 7), ("libero_goal", 3),
}
UNCHANGED = {  # the procedures that had stages when only 30 tasks were scored
    ("libero_90", 8): "(open wooden_cabinet_1_top_region) (Lifted akita_black_bowl_1)",
    ("libero_90", 21): "(turnon flat_stove_1) (Lifted chefmate_8_frypan_1)",
    ("libero_90", 50): "(Lifted alphabet_soup_1)",
    ("libero_90", 64): "(on akita_black_bowl_2 akita_black_bowl_1)",
    ("libero_90", 78): "(Lifted black_book_1)",
    ("libero_90", 84): "(Lifted red_coffee_mug_1)",
    ("libero_90", 88): "(Lifted yellow_book_1)",
    ("libero_10", 0): "(Or (Lifted alphabet_soup_1) (Lifted tomato_sauce_1)) "
                      "(Or (in alphabet_soup_1 basket_1_contain_region) (in tomato_sauce_1 basket_1_contain_region))",
    ("libero_10", 1): "(Or (Lifted cream_cheese_1) (Lifted butter_1)) "
                      "(Or (in cream_cheese_1 basket_1_contain_region) (in butter_1 basket_1_contain_region))",
    ("libero_10", 2): "(turnon flat_stove_1)",
    ("libero_10", 3): "(in akita_black_bowl_1 white_cabinet_1_bottom_region)",
    ("libero_10", 4): "(Or (on porcelain_mug_1 plate_1) (on white_yellow_mug_1 plate_2))",
    ("libero_10", 5): "(Lifted black_book_1)",
    ("libero_10", 6): "(Or (on porcelain_mug_1 plate_1) (on chocolate_pudding_1 living_room_table_plate_right_region))",
    ("libero_10", 7): "(Or (Lifted alphabet_soup_1) (Lifted cream_cheese_1)) "
                      "(Or (in alphabet_soup_1 basket_1_contain_region) (in cream_cheese_1 basket_1_contain_region))",
    ("libero_10", 8): "(Or (on moka_pot_1 flat_stove_1_cook_region) (on moka_pot_2 flat_stove_1_cook_region))",
    ("libero_10", 9): "(in white_yellow_mug_1 microwave_1_heating_region)",
    ("libero_spatial", 9): "(CarriedOff akita_black_bowl_1 wooden_cabinet_1)",
    ("libero_goal", 2): "(Lifted wine_bottle_1)",
    ("libero_goal", 4): "(Lifted akita_black_bowl_1)",
}
SUITE_SIZES = {"libero_90": 90, "libero_10": 10, "libero_spatial": 10, "libero_object": 10, "libero_goal": 10}
ALL_TASKS = [(suite, task_id) for suite in SUITES for task_id in range(SUITE_SIZES[suite])]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    # Import LIBERO only after writing an isolated config (no first-run prompt).
    config_dir = tmp_path_factory.mktemp("libero-procedural-score")
    config = {
        "benchmark_root": str(BENCHMARK_ROOT),
        "bddl_files": str(BENCHMARK_ROOT / "bddl_files"),
        "init_states": str(BENCHMARK_ROOT / "init_files"),
        "assets": str(BENCHMARK_ROOT / "assets"),
        "datasets": str(config_dir),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("LIBERO_CONFIG_PATH", str(config_dir))
        from libero.libero import benchmark
        from libero.libero.envs import procedural_score
        from libero.libero.envs.env_wrapper import ControlEnv

        yield benchmark, ControlEnv, procedural_score


@pytest.fixture()
def ps(runtime):
    return runtime[2]


# ----------------------------------------------------------------------------- scoring rules (stub env)


class Flag:
    """A condition read from a dict the test sets."""

    def __init__(self, key):
        self.key = key

    def bind(self, scorer):
        pass

    def __call__(self, scorer):
        return scorer.env.flags[self.key]


class StubEnv:
    class _Gripper:
        important_geoms = {"left_finger": [], "right_finger": []}
        contact_geoms = []

    class _Robot:
        def __init__(self):
            self.gripper = StubEnv._Gripper()
            self.robot_model = StubEnv._Gripper()

    def __init__(self, **flags):
        self.flags = {"a": False, "b": False, **flags}
        self.success = False
        self.robots = [self._Robot()]

    def _check_success(self):
        return self.success


@pytest.fixture()
def two_stages(ps, monkeypatch):
    monkeypatch.setitem(ps.PROCEDURES, "stub", (Flag("a"), Flag("b")))


def step(scorer, **flags):
    scorer.env.flags.update(flags)
    return scorer.update()


def test_stages_count_in_order_once_each(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    assert scorer.score == 0.0
    assert step(scorer, b=True) == 0.0                 # stage 2 before stage 1: no credit
    assert step(scorer, b=False, a=True) == 0.2
    assert step(scorer, a=False) == 0.2                # latched
    assert step(scorer, a=True) == 0.2                 # the same stage again: no more credit
    assert step(scorer, b=True) == 0.4
    assert step(scorer, a=False, b=False) == 0.4


def test_two_stages_can_complete_in_one_step(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    assert step(scorer, a=True, b=True) == 0.4


def test_success_scores_one_and_replaces_stage_credit(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    step(scorer, a=True)
    scorer.env.success = True
    assert scorer.update() == 1.0
    scorer.env.success = False
    assert scorer.update() == 1.0                      # latched
    fresh = ps.ProceduralScore(StubEnv(), "stub")
    assert fresh.update(success=True) == 1.0           # the caller's goal check is used as given


def test_conditions_true_at_the_start_need_a_rising_edge(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(a=True), "stub")
    assert step(scorer) == 0.0
    assert step(scorer, a=False) == 0.0
    assert step(scorer, a=True) == 0.2


def test_unknown_tasks_score_success_only(ps):
    scorer = ps.ProceduralScore(StubEnv(), "not_a_libero_task")
    assert scorer.stages == () and scorer.update() == 0.0 and scorer.update(success=True) == 1.0


def test_state_round_trips(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    step(scorer, a=True)
    copy = ps.ProceduralScore(StubEnv(), "stub")
    copy.load_state_dict(scorer.state_dict())
    assert copy.state_dict() == scorer.state_dict() and step(copy, b=True) == 0.4


def test_a_state_saved_without_footprints_still_loads(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    scorer.load_state_dict({"success": False, "credited": 1, "armed": [True, True], "rest_z": {"can": 0.9}})
    assert scorer.score == 0.2 and scorer.rest_z == {"can": 0.9} and step(scorer, b=True) == 0.4


# ----------------------------------------------------------------------------- the procedures (real scenes)


def get_task(benchmark, suite, task_id):
    return benchmark.get_benchmark(suite)().get_task(task_id)


def test_every_task_has_its_procedure(runtime):
    benchmark, _, ps = runtime
    names = set()
    for suite, task_id in ALL_TASKS:
        task = get_task(benchmark, suite, task_id)
        names.add(task.name)
        assert task.name in ps.PROCEDURES, (suite, task_id)
        stages = ps.PROCEDURES[task.name]
        expected = 0 if (suite, task_id) in NO_STAGE_TASKS else 2 if (suite, task_id) in TWO_STAGE_TASKS else 1
        assert len(stages) == expected, (suite, task_id)
    assert len(names) == 130 and names == set(ps.PROCEDURES)


def test_the_first_procedures_are_unchanged(runtime):
    benchmark, _, ps = runtime
    for (suite, task_id), expected in UNCHANGED.items():
        stages = ps.PROCEDURES[get_task(benchmark, suite, task_id).name]
        assert " ".join(map(repr, stages)) == expected, (suite, task_id)


def load_env(benchmark, ControlEnv, suite, task_id):
    task = get_task(benchmark, suite, task_id)
    env = ControlEnv(bddl_file_name=str(BENCHMARK_ROOT / "bddl_files" / task.problem_folder / task.bddl_file),
                     use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, hard_reset=False)
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    states = torch.load(BENCHMARK_ROOT / "init_files" / task.problem_folder / task.init_states_file, **load_options)
    env.seed(0)
    env.reset()
    return task, env, states


def settle(env, steps=None):
    from libero.libero.envs.procedural_score import SETTLE_STEPS

    for _ in range(SETTLE_STEPS if steps is None else steps):
        env.step([0, 0, 0, 0, 0, 0, -1])


@pytest.mark.parametrize("suite,task_id", ALL_TASKS)
def test_no_stage_holds_in_the_saved_initial_states(runtime, suite, task_id):
    benchmark, ControlEnv, ps = runtime
    task, env, states = load_env(benchmark, ControlEnv, suite, task_id)
    try:
        for index in range(0, len(states), 5):
            env.set_init_state(states[index])
            settle(env)
            scorer = ps.ProceduralScore(env.env, task.name)  # also checks every named object and region exists
            assert all(scorer.armed) and scorer.score == 0.0, index
        # Idling earns nothing, even in the few saved states where an object keeps moving by itself.
        for _ in range(20):
            env.step([0, 0, 0, 0, 0, 0, -1])
            assert scorer.update() == 0.0
    finally:
        env.close()


class Arm:
    """Absolute world-frame OSC_POSE moves of the grip site, scoring after every env step."""

    def __init__(self, env, scorer):
        self.env, self.scorer = env, scorer
        robot = env.robots[0]
        robot.controller.use_delta = False
        site = env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")]
        self.position = np.array(site)
        # Pointing down, as the arm starts.
        from scipy.spatial.transform import Rotation
        matrix = env.sim.data.site_xmat[env.sim.model.site_name2id("gripper0_grip_site")].reshape(3, 3)
        self.rotvec = Rotation.from_matrix(matrix).as_rotvec()

    def move(self, target, gripper, steps=60):
        start = self.position.copy()
        for t in np.linspace(0, 1, steps):
            self.position = start + t * (np.asarray(target) - start)
            self.env.step(np.concatenate([self.position, self.rotvec, [gripper]]))
            self.scorer.update(self.env.check_success())


def test_a_scripted_pick_earns_the_lift_stage_only_when_lifted(runtime):
    benchmark, ControlEnv, ps = runtime
    task, env, states = load_env(benchmark, ControlEnv, "libero_90", 50)  # the alphabet soup into the basket
    try:
        env.set_init_state(states[0])
        settle(env)
        scorer = ps.ProceduralScore(env.env, task.name)
        lifted = scorer.stages[0]
        can = scorer.position("alphabet_soup_1")
        arm = Arm(env, scorer)
        arm.move(can + [0, 0, 0.15], -1, 80)
        arm.move(can + [0, 0, 0.02], -1, 80)
        arm.move(can + [0, 0, 0.02], 1, 30)
        assert scorer.grasped("alphabet_soup_1") and not lifted(scorer)
        arm.move(can + [0, 0, 0.05], 1, 40)                # raised about 3 cm: not yet
        assert scorer.score == 0.0
        arm.move(can + [0, 0, 0.12], 1, 60)
        assert lifted(scorer) and scorer.score == 0.2
        arm.move(can + [0, 0, 0.12], -1, 30)               # dropped: the credit stays
        settle(env, 40)
        assert scorer.update() == 0.2 and not lifted(scorer)
    finally:
        env.close()


def place(env, obj, position):
    """Put a free object at a world position, at rest."""
    joint = env.env.get_object(obj).joints[-1]
    pose = np.array(env.sim.data.get_joint_qpos(joint))
    pose[:3] = position
    env.sim.data.set_joint_qpos(joint, pose)
    env.sim.data.qvel[:] = 0
    env.sim.forward()


@pytest.mark.parametrize("suite,task_id,fixture", [
    ("libero_spatial", 7, "flat_stove_1"),      # the bowl starts on the stove
    ("libero_spatial", 4, "wooden_cabinet_1"),  # the bowl starts in the open top drawer
    ("libero_90", 36, "microwave_1"),           # the white bowl starts on the microwave
])
def test_carried_off_needs_a_grasp_outside_the_footprint(runtime, monkeypatch, suite, task_id, fixture):
    benchmark, ControlEnv, ps = runtime
    task, env, states = load_env(benchmark, ControlEnv, suite, task_id)
    try:
        env.set_init_state(states[0])
        settle(env)
        scorer = ps.ProceduralScore(env.env, task.name)
        carried = scorer.stages[0]
        assert isinstance(carried, ps.CarriedOff) and carried.fixture == fixture
        obj = carried.obj
        start = scorer.position(obj)
        (origin, rotation), low, high = scorer.footprints[fixture]
        local = rotation.T @ (start - origin)
        assert np.all(local[:2] > low + 0.03) and np.all(local[:2] < high - 0.03)  # starts well inside
        grasp = {"held": True}
        monkeypatch.setattr(scorer, "grasped", lambda name: grasp["held"])
        assert not carried(scorer)                                   # held, but still over the fixture
        outside = origin + rotation @ np.array([high[0] + 0.05, local[1], local[2] + 0.10])
        place(env, obj, outside)
        assert carried(scorer)                                       # held and past the edge
        grasp["held"] = False
        assert not carried(scorer)                                   # past the edge but not held (knocked off)
    finally:
        env.close()
