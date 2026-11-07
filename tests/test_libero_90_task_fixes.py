"""Regression checks for the task corrections in this fork.

Goal probes set object reference points directly. They check the success
conditions, not a policy's ability to execute a placement trajectory.
"""
import inspect
from pathlib import Path
import re

import numpy as np
import pytest
import torch
import yaml


BENCHMARK_ROOT = Path(__file__).resolve().parents[1] / "libero" / "libero"
EXPECTED_TASKS = {
    27: (
        "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
        "put the wine bottle on the upper shelf of the wine rack",
    ),
    40: (
        "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
        "put the frying pan in the upper compartment of the wooden shelving unit",
    ),
    84: (
        "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
        "pick up the red mug and place it to the right compartment of the caddy",
    ),
    85: (
        "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
        "pick up the white mug and place it to the right compartment of the caddy",
    ),
    86: (
        "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
        "pick up the book in the middle and place it in the upper compartment of the wooden shelving unit",
    ),
    88: (
        "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
        "pick up the book on the right and place it in the upper compartment of the wooden shelving unit",
    ),
}


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    # Import LIBERO only after writing an isolated config, to avoid its first-run
    # prompt and avoid changing an existing user's asset or dataset paths.
    config_dir = tmp_path_factory.mktemp("libero-task-fixes")
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
        from libero.libero.envs.env_wrapper import ControlEnv

        yield benchmark, ControlEnv


def test_task_ids_names_and_file_paths_are_preserved(runtime):
    benchmark, _ = runtime
    suite = benchmark.get_benchmark("libero_90")()
    assert suite.get_num_tasks() == 90
    for task_id, (name, instruction) in EXPECTED_TASKS.items():
        task = suite.get_task(task_id)
        assert task.name == name
        assert task.bddl_file == name + ".bddl"
        assert task.init_states_file == name + ".pruned_init"
        assert suite.get_task_demonstration(task_id) == "libero_90/" + name + "_demo.hdf5"
        assert task.language == instruction
        text = (BENCHMARK_ROOT / "bddl_files/libero_90" / task.bddl_file).read_text()
        assert re.search(r"\(:language\s+([^)]*)\)", text).group(1) == instruction


def test_only_the_requested_api_instructions_change(runtime):
    benchmark, _ = runtime
    changed_names = {entry[0] for entry in EXPECTED_TASKS.values()} | {
        "put_the_wine_bottle_on_the_rack",
    }
    count = 0
    for task_map in benchmark.task_maps.values():
        for task in task_map.values():
            count += 1
            if task.name not in changed_names:
                assert task.language == benchmark.grab_language_from_filename(task.bddl_file)
    assert count == 130


@pytest.fixture(scope="module", params=[40, 84, 85, 86, 88])
def task_environment(request, runtime):
    benchmark, ControlEnv = runtime
    task_id = request.param
    task = benchmark.get_benchmark("libero_90")().get_task(task_id)
    env = ControlEnv(
        bddl_file_name=str(BENCHMARK_ROOT / "bddl_files/libero_90" / task.bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    env.seed(0)
    env.reset()
    path = BENCHMARK_ROOT / "init_files/libero_90" / task.init_states_file
    load_options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    states = torch.load(path, **load_options)
    env.set_init_state(states[0])
    try:
        yield task_id, task, env, states
    finally:
        env.close()


def test_benchmark_and_environment_instructions_agree(task_environment):
    task_id, task, env, _ = task_environment
    assert task.language == env.language_instruction == EXPECTED_TASKS[task_id][1]


def test_all_saved_initial_states_load_without_already_completing_task(task_environment):
    _, _, env, states = task_environment
    assert len(states) == 50
    for index, state in enumerate(states):
        assert np.isfinite(state).all(), index
        env.set_init_state(state)
        assert not env.check_success(), index


def test_goal_accepts_requested_compartment_and_rejects_alternatives(task_environment):
    task_id, _, env, states = task_environment
    env.set_init_state(states[0])
    domain = env.env
    goals = domain.parsed_problem["goal_state"]
    assert len(goals) == 1
    predicate, object_name, target = goals[0]
    assert predicate == "in"
    if task_id in (84, 85):
        assert target == "desk_caddy_1_right_contain_region"
        rejected_regions = [
            "desk_caddy_1_left_contain_region",
            "desk_caddy_1_front_contain_region",
            "desk_caddy_1_back_contain_region",
            "study_table_desk_caddy_right_region",
        ]
    else:
        assert target == "wooden_two_layer_shelf_1_top_region"
        rejected_regions = [
            "wooden_two_layer_shelf_1_bottom_region",
            "wooden_two_layer_shelf_1_top_side",
        ]
    obj = domain.get_object(object_name)
    original_pose = np.array(env.sim.data.get_joint_qpos(obj.joints[-1]), copy=True)

    def accepts(point):
        pose = original_pose.copy()
        pose[:3] = point
        env.sim.data.set_joint_qpos(obj.joints[-1], pose)
        env.sim.forward()
        return env.check_success()

    assert accepts(env.sim.data.get_site_xpos(target).copy())
    for region in rejected_regions:
        point = env.sim.data.get_site_xpos(region).copy()
        if region.endswith("_top_side"):
            point[2] += 0.03
        elif region.startswith("study_table_"):
            point[2] += 0.04
        assert not accepts(point), region


@pytest.mark.parametrize("task_id", [84, 85])
def test_caddy_tasks_keep_their_bddl_instructions(task_id):
    name, instruction = EXPECTED_TASKS[task_id]
    text = (BENCHMARK_ROOT / "bddl_files/libero_90" / (name + ".bddl")).read_text()
    assert re.search(r"\(:language\s+([^)]*)\)", text).group(1) == instruction


@pytest.mark.parametrize("task_id, instruction, goal", [
    (17, "stack the black bowl in the middle on the black bowl at the back",
     "(On akita_black_bowl_2 akita_black_bowl_3)"),
    (27, EXPECTED_TASKS[27][1], "(On wine_bottle_1 wine_rack_1_top_region)"),
    (86, EXPECTED_TASKS[86][1], "(In black_book_1 wooden_two_layer_shelf_1_top_region)"),
])
def test_instruction_fixes_preserve_the_goal(runtime, task_id, instruction, goal):
    benchmark, _ = runtime
    task = benchmark.get_benchmark("libero_90")().get_task(task_id)
    text = (BENCHMARK_ROOT / "bddl_files/libero_90" / task.bddl_file).read_text()
    assert re.search(r"\(:language\s+([^)]*)\)", text).group(1) == instruction
    assert text[text.index("(:goal"):].split() == ("(:goal (And " + goal + ") ) )").split()
    if task_id == 17:
        assert task.name == "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl"
        assert task.language == "stack the middle black bowl on the back black bowl"


@pytest.mark.parametrize("class_name", ["WoodenCabinet", "WhiteCabinet"])
def test_cabinet_closed_boundary(runtime, class_name):
    from libero.libero.envs.objects import articulated_objects

    cabinet = getattr(articulated_objects, class_name)()
    for qpos in (-0.15, -0.005, -1e-9):
        assert not cabinet.is_close(qpos), qpos
    for qpos in (-0.0, 0.0, 0.0025, 0.005):
        assert cabinet.is_close(qpos), qpos


@pytest.mark.parametrize("task_id", [0, 22, 28])
def test_closed_drawer_at_zero_passes_native_goal_after_settling(runtime, task_id):
    benchmark, ControlEnv = runtime
    suite = benchmark.get_benchmark("libero_90")()
    env = ControlEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )
    try:
        env.seed(0)
        env.reset()
        domain = env.env
        predicate, target = domain.parsed_problem["goal_state"][0]
        assert predicate == "close"
        drawer = domain.object_states_dict[target]
        drawer.set_joint(-0.15)
        env.sim.forward()
        assert not env.check_success()
        drawer.set_joint(0.0)
        env.sim.data.qvel[:] = 0.0
        env.sim.forward()
        assert env.check_success()
        for _ in range(60):
            env.step(np.zeros(7))
        assert env.check_success()
    finally:
        env.close()
