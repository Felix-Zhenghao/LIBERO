"""Procedural scores: partial credit for progress on selected LIBERO tasks.

A task's procedure is an ordered list of stages followed by the task's own goal
(its BDDL success check). Each stage completed in order earns `STAGE_CREDIT`;
completing the task scores 1, which replaces any stage credit. So a task never
scores more than 1.

- Stages count in order: a stage is credited only after the previous one.
- Each stage is credited at most once, however often its condition holds again
  (another object lifted, the same object lifted twice).
- A stage is credited only on a rising edge: its condition must have been seen
  false earlier in the episode, so nothing is credited for the initial scene.

Stage conditions reuse the BDDL goal predicates where they can (`Goal`). The new
ones are strict on purpose:

- `Lifted(obj)`: both fingers touch the object, nothing but the robot touches it,
  and it is at least `LIFT_HEIGHT` above where it rested when the tracker was
  created (after the scene settled).
- `CarriedOff(obj, fixture)`: both fingers touch the object, and its reference
  point is outside the fixture's footprint, the vertical prism over the
  fixture's collision geometry in the fixture's own frame.

Tasks without an entry in `PROCEDURES` score 1 on success and 0 otherwise.

Usage, with `env` the BDDL problem env (`OffScreenRenderEnv(...).env`):

    score = ProceduralScore(env, task.name)   # after reset and settling
    score.update()                            # after every env step
    score.score                               # 0, 0.2, 0.4, ... or 1
    state = score.state_dict(); score.load_state_dict(state)
"""

import numpy as np

STAGE_CREDIT = 0.2
LIFT_HEIGHT = 0.05  # metres


# ----------------------------------------------------------------------------- stage conditions


# A condition is stateless and shared by every tracker: `bind` stores what it measures at the start of an
# episode on the tracker, and `__call__` evaluates it on the tracker's env.


class Goal:
    """A BDDL predicate, evaluated as the task's goal check evaluates it."""

    def __init__(self, *state):
        self.state = list(state)

    def bind(self, scorer):
        pass

    def __call__(self, scorer):
        return bool(scorer.env._eval_predicate(self.state))

    def __repr__(self):
        return f"({' '.join(self.state)})"


class AnyOf:
    """True when any of the conditions holds."""

    def __init__(self, *conditions):
        self.conditions = conditions

    def bind(self, scorer):
        for condition in self.conditions:
            condition.bind(scorer)

    def __call__(self, scorer):
        return any(condition(scorer) for condition in self.conditions)

    def __repr__(self):
        return f"(Or {' '.join(map(repr, self.conditions))})"


class Lifted:
    """Grasped, supported only by the robot, and at least LIFT_HEIGHT above its rest height."""

    def __init__(self, obj):
        self.obj = obj

    def bind(self, scorer):
        scorer.rest_z.setdefault(self.obj, float(scorer.position(self.obj)[2]))

    def __call__(self, scorer):
        return (
            scorer.position(self.obj)[2] - scorer.rest_z[self.obj] >= LIFT_HEIGHT
            and scorer.grasped(self.obj)
            and scorer.held_only_by_robot(self.obj)
        )

    def __repr__(self):
        return f"(Lifted {self.obj})"


class CarriedOff:
    """Grasped, with its reference point outside the fixture's footprint."""

    def __init__(self, obj, fixture):
        self.obj, self.fixture = obj, fixture

    def bind(self, scorer):
        scorer.footprints.setdefault(self.fixture, scorer.footprint(self.fixture))

    def __call__(self, scorer):
        (origin, rotation), low, high = scorer.footprints[self.fixture]
        local = rotation.T @ (scorer.position(self.obj) - origin)
        inside = np.all(local[:2] >= low) and np.all(local[:2] <= high)
        return (not inside) and scorer.grasped(self.obj)

    def __repr__(self):
        return f"(CarriedOff {self.obj} {self.fixture})"


# ----------------------------------------------------------------------------- the procedures

LIBERO_BASKET = "basket_1_contain_region"
STOVE_TOP = "flat_stove_1_cook_region"


def _two_into_basket(a, b):
    return (
        AnyOf(Lifted(a), Lifted(b)),
        AnyOf(Goal("in", a, LIBERO_BASKET), Goal("in", b, LIBERO_BASKET)),
    )


# Keyed by LIBERO task name (the BDDL file name); the stages precede the task's own goal.
PROCEDURES = {
    # LIBERO-90: the research plan's training tasks
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it": (     # 8
        Goal("open", "wooden_cabinet_1_top_region"),
        Lifted("akita_black_bowl_1"),
    ),
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it": (                 # 21
        Goal("turnon", "flat_stove_1"),
        Lifted("chefmate_8_frypan_1"),
    ),
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate": (),                 # 37
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf": (),                      # 40
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket": (         # 50
        Lifted("alphabet_soup_1"),
    ),
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray": (  # 64
        Goal("on", "akita_black_bowl_2", "akita_black_bowl_1"),
    ),
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate": (),          # 68
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": (  # 78
        Lifted("black_book_1"),
    ),
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy": (       # 84
        Lifted("red_coffee_mug_1"),
    ),
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf": (  # 88
        Lifted("yellow_book_1"),
    ),
    # LIBERO-10
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket":  # 0
        _two_into_basket("alphabet_soup_1", "tomato_sauce_1"),
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket":     # 1
        _two_into_basket("cream_cheese_1", "butter_1"),
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": (                   # 2
        Goal("turnon", "flat_stove_1"),
    ),
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": (  # 3
        Goal("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
    ),
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": (  # 4
        AnyOf(Goal("on", "porcelain_mug_1", "plate_1"), Goal("on", "white_yellow_mug_1", "plate_2")),
    ),
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": (  # 5
        Lifted("black_book_1"),
    ),
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": (  # 6
        AnyOf(
            Goal("on", "porcelain_mug_1", "plate_1"),
            Goal("on", "chocolate_pudding_1", "living_room_table_plate_right_region"),
        ),
    ),
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket":  # 7
        _two_into_basket("alphabet_soup_1", "cream_cheese_1"),
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (                                # 8
        AnyOf(Goal("on", "moka_pot_1", STOVE_TOP), Goal("on", "moka_pot_2", STOVE_TOP)),
    ),
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": (     # 9
        Goal("in", "white_yellow_mug_1", "microwave_1_heating_region"),
    ),
    # LIBERO-Spatial
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": (),         # 1
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": (),           # 3
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": (),              # 5
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (),                # 7
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": (         # 9
        CarriedOff("akita_black_bowl_1", "wooden_cabinet_1"),
    ),
    # LIBERO-Goal
    "open_the_middle_drawer_of_the_cabinet": (),                                        # 0
    "put_the_wine_bottle_on_top_of_the_cabinet": (Lifted("wine_bottle_1"),),            # 2
    "put_the_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_1"),),              # 4
    "put_the_cream_cheese_in_the_bowl": (),                                             # 6
    "put_the_bowl_on_the_plate": (),                                                    # 8
}


# ----------------------------------------------------------------------------- the tracker


class ProceduralScore:
    """The procedural score of one episode, latched across env steps."""

    def __init__(self, env, task_name):
        self.env = env
        self.task_name = task_name
        self.stages = PROCEDURES.get(task_name, ())
        gripper = env.robots[0].gripper
        self._fingers = [gripper.important_geoms["left_finger"], gripper.important_geoms["right_finger"]]
        self._robot_geoms = set(gripper.contact_geoms) | set(env.robots[0].robot_model.contact_geoms)
        self.rest_z = {}       # object -> its height at rest (Lifted)
        self.footprints = {}   # fixture -> its footprint (CarriedOff)
        for stage in self.stages:
            stage.bind(self)
        self.success = False
        self.credited = 0                       # stages credited so far (in order)
        self.armed = [False] * len(self.stages)  # condition seen false since the episode started
        self.update()

    # ------------------------------------------------------------------ scoring

    @property
    def score(self):
        return 1.0 if self.success else round(STAGE_CREDIT * self.credited, 6)

    def update(self, success=None):
        """Evaluate the current state; `success` is the task's goal check if the caller already has it."""
        if not self.success:
            self.success = bool(self.env._check_success() if success is None else success)
        if not self.success:
            for i, stage in enumerate(self.stages):
                if not stage(self):
                    self.armed[i] = True
                elif self.armed[i] and i == self.credited:
                    self.credited += 1
        return self.score

    def state_dict(self):
        return {"success": self.success, "credited": self.credited, "armed": list(self.armed),
                "rest_z": dict(self.rest_z)}

    def load_state_dict(self, state):
        self.success = bool(state["success"])
        self.credited = int(state["credited"])
        self.armed = [bool(v) for v in state["armed"]]
        self.rest_z.update({k: float(v) for k, v in state["rest_z"].items()})

    # ------------------------------------------------------------------ geometry and contacts

    def position(self, obj):
        return np.array(self.env.sim.data.body_xpos[self.env.obj_body_id[obj]])

    def grasped(self, obj):
        """Both fingers (pad or finger body) touch the object."""
        return self.env._check_grasp(self._fingers, self.env.get_object(obj))

    def held_only_by_robot(self, obj):
        """Nothing but the robot touches the object (unnamed geoms count as not the robot)."""
        return self.env.get_contacts(self.env.get_object(obj)) <= self._robot_geoms

    def footprint(self, fixture):
        """((origin, rotation), low, high): the fixture's collision geometry's xy extent in its own frame."""
        sim = self.env.sim
        body = self.env.obj_body_id[fixture]
        origin = np.array(sim.data.body_xpos[body])
        rotation = np.array(sim.data.body_xmat[body]).reshape(3, 3)
        corners = []
        for name in self.env.get_object(fixture).contact_geoms:
            geom = sim.model.geom_name2id(name)
            center, half = sim.model.geom_aabb[geom][:3], sim.model.geom_aabb[geom][3:]
            geom_rotation = np.array(sim.data.geom_xmat[geom]).reshape(3, 3)
            for signs in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                world = sim.data.geom_xpos[geom] + geom_rotation @ (center + half * signs)
                corners.append(rotation.T @ (world - origin))
        corners = np.array(corners)
        return (origin, rotation), corners[:, :2].min(0), corners[:, :2].max(0)
