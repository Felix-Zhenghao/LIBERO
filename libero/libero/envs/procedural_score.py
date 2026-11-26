"""Procedural scores: partial credit for progress on every LIBERO task.

A task's procedure is an ordered list of stages followed by the task's own goal
(its BDDL success check). Each stage completed in order earns `STAGE_CREDIT`;
completing the task scores 1, which replaces any stage credit. So a task never
scores more than 1.

- Stages count in order: a stage is credited only after the previous one.
- Each stage is credited at most once, however often its condition holds again
  (another object lifted, the same object lifted twice).
- A stage is credited only on a rising edge: its condition must have been seen
  false earlier in the episode, so nothing is credited for the initial scene.

Stage conditions reuse the BDDL goal predicates where they can (`Goal`). The
others are strict on purpose:

- `Lifted(obj)`: both fingers touch the object, nothing but the robot touches it,
  and it is at least `LIFT_HEIGHT` above where it rested when the tracker was
  created (after the scene settled).
- `CarriedOff(obj, fixture)`: both fingers touch the object, and its reference
  point is outside the fixture's footprint, the vertical prism over the
  fixture's collision geometry in the fixture's own frame.

All 130 LIBERO tasks have an entry in `PROCEDURES`. A task with no stages, or a
task name without an entry, scores 1 on success and 0 otherwise.

Create the tracker only after the scene has settled. LIBERO's saved initial
states are not at rest: some objects are pushed up out of the table and land
again a few control steps later. Step the env with zero actions for
`SETTLE_STEPS` steps first, or `Lifted` measures a rest height that is too high.

Usage, with `env` the BDDL problem env (`OffScreenRenderEnv(...).env`):

    score = ProceduralScore(env, task.name)   # after reset and SETTLE_STEPS zero actions
    score.update()                            # after every env step
    score.score                               # 0, 0.2, 0.4, ... or 1
    state = score.state_dict(); score.load_state_dict(state)
"""

import numpy as np

STAGE_CREDIT = 0.2
LIFT_HEIGHT = 0.05  # metres
SETTLE_STEPS = 10   # zero-action control steps after reset, before the tracker is created


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


# Keyed by LIBERO task name (the BDDL file name); the stages precede the task's own goal. The number after
# each entry is the task's id in its suite. How each procedure was chosen:
#
# - A task with several clauses credits each finished clause, in the instruction's order, with the BDDL
#   predicate of that clause.
# - A task that moves an object credits the pick-up before the placement: `CarriedOff` when the object
#   starts on or in a fixture it has to leave, `Lifted` otherwise.
# - A task that is one continuous motion (a drawer, a door, a knob, a push, a slide) has no stage: BDDL has
#   no intermediate state for it, so it scores success only.
#
# The 20 procedures that already had stages in the first 30 tasks are unchanged.
PROCEDURES = {
    # ----------------------------------------------------------------------------------------------------
    # LIBERO-90
    # KITCHEN_SCENE10: wooden cabinet with the top drawer open, a black bowl, two butters, a chocolate pudding
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet": (),  # 0
    "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it": (  # 1
        Goal("close", "wooden_cabinet_1_top_region"),
        Lifted("akita_black_bowl_1"),
    ),
    "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 2
    "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it": (  # 3
        Lifted("butter_2"),
        Goal("in", "butter_2", "wooden_cabinet_1_top_region"),
    ),
    "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it": (  # 4
        Lifted("butter_1"),
        Goal("in", "butter_1", "wooden_cabinet_1_top_region"),
    ),
    "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it": (  # 5
        Lifted("chocolate_pudding_1"),
        Goal("in", "chocolate_pudding_1", "wooden_cabinet_1_top_region"),
    ),
    # KITCHEN_SCENE1: wooden cabinet (closed), a black bowl, a plate
    "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet": (),  # 6
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet": (),  # 7
    "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it": (  # 8
        Goal("open", "wooden_cabinet_1_top_region"),
        Lifted("akita_black_bowl_1"),
    ),
    "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 9
    "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 10
    # KITCHEN_SCENE2: wooden cabinet (closed), three black bowls, a plate
    "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet": (),  # 11
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate": (Lifted("akita_black_bowl_3"),),  # 12
    "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 13
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate": (Lifted("akita_black_bowl_2"),),  # 14
    "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_2"),),  # 15
    "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle": (Lifted("akita_black_bowl_1"),),  # 16
    "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl": (Lifted("akita_black_bowl_2"),),  # 17
    # KITCHEN_SCENE3: stove (off), a frying pan, a moka pot
    "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove": (Lifted("chefmate_8_frypan_1"),),  # 18
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove": (Lifted("moka_pot_1"),),  # 19
    "KITCHEN_SCENE3_turn_on_the_stove": (),  # 20
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it": (  # 21
        Goal("turnon", "flat_stove_1"),
        Lifted("chefmate_8_frypan_1"),
    ),
    # KITCHEN_SCENE4: white cabinet with the bottom drawer open, a wine rack, a black bowl, a wine bottle
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet": (),  # 22
    "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer": (Goal("close", "white_cabinet_1_bottom_region"),),  # 23
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 24
    "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 25
    "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet": (Lifted("wine_bottle_1"),),  # 26
    "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack": (Lifted("wine_bottle_1"),),  # 27
    # KITCHEN_SCENE5: white cabinet with the top drawer open, a black bowl, a ketchup, a plate
    "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet": (),  # 28
    "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 29
    "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 30
    "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 31
    "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet": (Lifted("ketchup_1"),),  # 32
    # KITCHEN_SCENE6: microwave (open), a white mug, a yellow and white mug
    "KITCHEN_SCENE6_close_the_microwave": (),  # 33
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug": (Lifted("white_yellow_mug_1"),),  # 34
    # KITCHEN_SCENE7: microwave (closed) with a white bowl on top, a plate
    "KITCHEN_SCENE7_open_the_microwave": (),  # 35
    "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate": (CarriedOff("white_bowl_1", "microwave_1"),),  # 36
    "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate": (CarriedOff("white_bowl_1", "microwave_1"),),  # 37
    # KITCHEN_SCENE8: stove (on), two moka pots
    "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove": (Lifted("moka_pot_1"),),  # 38
    "KITCHEN_SCENE8_turn_off_the_stove": (),  # 39
    # KITCHEN_SCENE9: stove (off), a two-layer shelf, a frying pan, a white bowl
    "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf": (Lifted("chefmate_8_frypan_1"),),  # 40
    "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet": (Lifted("chefmate_8_frypan_1"),),  # 41
    "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf": (),  # 42
    "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet": (Lifted("white_bowl_1"),),  # 43
    "KITCHEN_SCENE9_turn_on_the_stove": (),  # 44
    "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it": (  # 45
        Goal("turnon", "flat_stove_1"),
        Lifted("chefmate_8_frypan_1"),
    ),
    # LIVING_ROOM_SCENE1: basket and four groceries
    "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket": (Lifted("alphabet_soup_1"),),  # 46
    "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket": (Lifted("cream_cheese_1"),),  # 47
    "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket": (Lifted("ketchup_1"),),  # 48
    "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket": (Lifted("tomato_sauce_1"),),  # 49
    # LIVING_ROOM_SCENE2: basket and seven groceries
    "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket": (Lifted("alphabet_soup_1"),),  # 50
    "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket": (Lifted("butter_1"),),  # 51
    "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket": (Lifted("milk_1"),),  # 52
    "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket": (Lifted("orange_juice_1"),),  # 53
    "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket": (Lifted("tomato_sauce_1"),),  # 54
    # LIVING_ROOM_SCENE3: wooden tray and five groceries
    "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray": (Lifted("alphabet_soup_1"),),  # 55
    "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray": (Lifted("butter_1"),),  # 56
    "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray": (Lifted("cream_cheese_1"),),  # 57
    "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray": (Lifted("ketchup_1"),),  # 58
    "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray": (Lifted("tomato_sauce_1"),),  # 59
    # LIVING_ROOM_SCENE4: wooden tray, two black bowls, a chocolate pudding, a salad dressing
    "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray": (Lifted("akita_black_bowl_1"),),  # 60
    "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray": (Lifted("chocolate_pudding_1"),),  # 61
    "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray": (Lifted("new_salad_dressing_1"),),  # 62
    "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray": (  # 63
        Goal("on", "akita_black_bowl_1", "akita_black_bowl_2"),
    ),
    "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray": (  # 64
        Goal("on", "akita_black_bowl_2", "akita_black_bowl_1"),
    ),
    # LIVING_ROOM_SCENE5: two plates, three mugs
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate": (Lifted("red_coffee_mug_1"),),  # 65
    "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate": (Lifted("red_coffee_mug_1"),),  # 66
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate": (Lifted("porcelain_mug_1"),),  # 67
    "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate": (Lifted("white_yellow_mug_1"),),  # 68
    # LIVING_ROOM_SCENE6: a plate, two mugs, a chocolate pudding
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate": (Lifted("chocolate_pudding_1"),),  # 69
    "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate": (Lifted("chocolate_pudding_1"),),  # 70
    "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate": (Lifted("red_coffee_mug_1"),),  # 71
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate": (Lifted("porcelain_mug_1"),),  # 72
    # STUDY_SCENE1: desk caddy, a book, a yellow and white mug
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 73
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 74
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 75
    "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy": (Lifted("white_yellow_mug_1"),),  # 76
    # STUDY_SCENE2: desk caddy, a book, a red mug
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 77
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 78
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 79
    "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 80
    # STUDY_SCENE3: desk caddy, a book, a white mug, a red mug
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 81
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 82
    "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 83
    "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy": (Lifted("red_coffee_mug_1"),),  # 84
    "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy": (Lifted("porcelain_mug_1"),),  # 85
    # STUDY_SCENE4: two-layer shelf, three books
    "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf": (Lifted("black_book_1"),),  # 86
    "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf": (Lifted("yellow_book_2"),),  # 87
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf": (Lifted("yellow_book_1"),),  # 88
    "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf": (Lifted("yellow_book_1"),),  # 89
    # ----------------------------------------------------------------------------------------------------
    # LIBERO-10 (unchanged from the first 30)
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket":  # 0
        _two_into_basket("alphabet_soup_1", "tomato_sauce_1"),
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket":  # 1
        _two_into_basket("cream_cheese_1", "butter_1"),
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it": (Goal("turnon", "flat_stove_1"),),  # 2
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it": (  # 3
        Goal("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
    ),
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": (  # 4
        AnyOf(Goal("on", "porcelain_mug_1", "plate_1"), Goal("on", "white_yellow_mug_1", "plate_2")),
    ),
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy": (Lifted("black_book_1"),),  # 5
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate": (  # 6
        AnyOf(
            Goal("on", "porcelain_mug_1", "plate_1"),
            Goal("on", "chocolate_pudding_1", "living_room_table_plate_right_region"),
        ),
    ),
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket":  # 7
        _two_into_basket("alphabet_soup_1", "cream_cheese_1"),
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (AnyOf(Goal("on", "moka_pot_1", STOVE_TOP), Goal("on", "moka_pot_2", STOVE_TOP)),),  # 8
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it": (  # 9
        Goal("in", "white_yellow_mug_1", "microwave_1_heating_region"),
    ),
    # ----------------------------------------------------------------------------------------------------
    # LIBERO-Spatial: two black bowls; the goal is always (On akita_black_bowl_1 plate_1)
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 0
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 1
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 2
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 3
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": (  # 4
        CarriedOff("akita_black_bowl_1", "wooden_cabinet_1"),
    ),
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 5
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 6
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (CarriedOff("akita_black_bowl_1", "flat_stove_1"),),  # 7
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 8
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": (CarriedOff("akita_black_bowl_1", "wooden_cabinet_1"),),  # 9
    # ----------------------------------------------------------------------------------------------------
    # LIBERO-Object: the goal is always (In <object> basket_1_contain_region)
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket": (Lifted("alphabet_soup_1"),),  # 0
    "pick_up_the_cream_cheese_and_place_it_in_the_basket": (Lifted("cream_cheese_1"),),  # 1
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": (Lifted("salad_dressing_1"),),  # 2
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket": (Lifted("bbq_sauce_1"),),  # 3
    "pick_up_the_ketchup_and_place_it_in_the_basket": (Lifted("ketchup_1"),),  # 4
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket": (Lifted("tomato_sauce_1"),),  # 5
    "pick_up_the_butter_and_place_it_in_the_basket": (Lifted("butter_1"),),  # 6
    "pick_up_the_milk_and_place_it_in_the_basket": (Lifted("milk_1"),),  # 7
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket": (Lifted("chocolate_pudding_1"),),  # 8
    "pick_up_the_orange_juice_and_place_it_in_the_basket": (Lifted("orange_juice_1"),),  # 9
    # ----------------------------------------------------------------------------------------------------
    # LIBERO-Goal: one scene (cabinet, stove, wine rack, bowl, plate, cream cheese, wine bottle)
    "open_the_middle_drawer_of_the_cabinet": (),  # 0
    "put_the_bowl_on_the_stove": (Lifted("akita_black_bowl_1"),),  # 1
    "put_the_wine_bottle_on_top_of_the_cabinet": (Lifted("wine_bottle_1"),),  # 2
    "open_the_top_drawer_and_put_the_bowl_inside": (  # 3
        Goal("open", "wooden_cabinet_1_top_region"),
        Lifted("akita_black_bowl_1"),
    ),
    "put_the_bowl_on_top_of_the_cabinet": (Lifted("akita_black_bowl_1"),),  # 4
    "push_the_plate_to_the_front_of_the_stove": (),  # 5
    "put_the_cream_cheese_in_the_bowl": (Lifted("cream_cheese_1"),),  # 6
    "turn_on_the_stove": (),  # 7
    "put_the_bowl_on_the_plate": (Lifted("akita_black_bowl_1"),),  # 8
    "put_the_wine_bottle_on_the_rack": (Lifted("wine_bottle_1"),),  # 9
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
        self.rest_z = {}          # object -> its height at rest (Lifted)
        self.footprints = {}      # fixture -> its footprint (CarriedOff)
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
        footprints = {
            fixture: [origin.tolist(), rotation.tolist(), low.tolist(), high.tolist()]
            for fixture, ((origin, rotation), low, high) in self.footprints.items()
        }
        return {"success": self.success, "credited": self.credited, "armed": list(self.armed),
                "rest_z": dict(self.rest_z), "footprints": footprints}

    def load_state_dict(self, state):
        self.success = bool(state["success"])
        self.credited = int(state["credited"])
        self.armed = [bool(v) for v in state["armed"]]
        self.rest_z.update({k: float(v) for k, v in state["rest_z"].items()})
        for fixture, (origin, rotation, low, high) in state.get("footprints", {}).items():  # absent from older saved states
            self.footprints[fixture] = ((np.array(origin), np.array(rotation)), np.array(low), np.array(high))

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
