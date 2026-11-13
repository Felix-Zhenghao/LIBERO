"""Checks for the procedural scores (libero/libero/envs/procedural_score.py).

The scoring rules run on a stub env. The procedures are checked against the
research plan's task list, the BDDL objects, and the saved initial states. A
scripted pick in a real scene checks that `Lifted` holds only for a real lift.
"""
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "libero" / "libero"
PLAN_TASKS = {  # the research plan's training and validation tasks
    "libero_90": [8, 21, 37, 40, 50, 64, 68, 78, 84, 88],
    "libero_10": list(range(10)),
    "libero_spatial": [1, 3, 5, 7, 9],
    "libero_goal": [0, 2, 4, 6, 8],
}
STAGE_COUNTS = {
    ("libero_90", 8): 2, ("libero_90", 21): 2, ("libero_90", 50): 1, ("libero_90", 64): 1, ("libero_90", 78): 1,
    ("libero_90", 84): 1, ("libero_90", 88): 1,
    ("libero_10", 0): 2, ("libero_10", 1): 2, ("libero_10", 2): 1, ("libero_10", 3): 1, ("libero_10", 4): 1,
    ("libero_10", 5): 1, ("libero_10", 6): 1, ("libero_10", 7): 2, ("libero_10", 8): 1, ("libero_10", 9): 1,
    ("libero_spatial", 9): 1, ("libero_goal", 2): 1, ("libero_goal", 4): 1,
}  # every other plan task scores success only


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
    scorer = ps.ProceduralScore(StubEnv(), "not_a_planned_task")
    assert scorer.stages == () and scorer.update() == 0.0 and scorer.update(success=True) == 1.0


def test_state_round_trips(ps, two_stages):
    scorer = ps.ProceduralScore(StubEnv(), "stub")
    step(scorer, a=True)
    copy = ps.ProceduralScore(StubEnv(), "stub")
    copy.load_state_dict(scorer.state_dict())
    assert copy.state_dict() == scorer.state_dict() and step(copy, b=True) == 0.4


# ----------------------------------------------------------------------------- the procedures (real scenes)


def plan_tasks(benchmark):
    for suite, ids in PLAN_TASKS.items():
        for task_id in ids:
            yield suite, task_id, benchmark.get_benchmark(suite)().get_task(task_id)


def test_every_plan_task_has_its_procedure(runtime):
    benchmark, _, ps = runtime
    names = set()
    for suite, task_id, task in plan_tasks(benchmark):
        names.add(task.name)
        assert task.name in ps.PROCEDURES, (suite, task_id)
        assert len(ps.PROCEDURES[task.name]) == STAGE_COUNTS.get((suite, task_id), 0), (suite, task_id)
    assert names == set(ps.PROCEDURES)


def load_env(benchmark, ControlEnv, suite, task_id):
    task = benchmark.get_benchmark(suite)().get_task(task_id)
    env = ControlEnv(bddl_file_name=str(BENCHMARK_ROOT / "bddl_files" / task.problem_folder / task.bddl_file),
                     use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, hard_reset=False)
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    states = torch.load(BENCHMARK_ROOT / "init_files" / task.problem_folder / task.init_states_file, **load_options)
    env.seed(0)
    env.reset()
    return task, env, states


def settle(env, steps=10):
    for _ in range(steps):
        env.step([0, 0, 0, 0, 0, 0, -1])


@pytest.mark.parametrize("suite,task_id", sorted(STAGE_COUNTS))
def test_no_stage_holds_in_the_saved_initial_states(runtime, suite, task_id):
    benchmark, ControlEnv, ps = runtime
    task, env, states = load_env(benchmark, ControlEnv, suite, task_id)
    try:
        for index in range(0, len(states), 5):
            env.set_init_state(states[index])
            settle(env)
            scorer = ps.ProceduralScore(env.env, task.name)  # also checks every named object and region exists
            assert all(scorer.armed) and scorer.score == 0.0, index
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
