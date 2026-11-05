"""Constants shared by publication analyses and figures."""

MODEL_ANIMAL_IDS = (
    "W051",
    "W060",
    "W065",
    "W066",
    "W068",
    "W072",
    "W073",
    "W074",
    "W075",
    "W078",
    "W080",
    "W081",
    "W083",
    "W088",
    "W089",
    "W094",
)

ALL_ANIMAL_IDS = tuple(sorted((*MODEL_ANIMAL_IDS, "W082")))

# Backward-compatible name for the cohort used in the manuscript GLMs.
ANIMAL_IDS = MODEL_ANIMAL_IDS

# Each is used by at least 13 of 16 trained animals; 5, 7, and 8 s by one or two.
TRAINED_DELAYS = (2.0, 3.0, 4.0, 6.0)

RANDOM_STATE = 47
TEST_SIZE = 0.2

HIT_COLOR = "lime"
VIOLATION_COLOR = "#e1c63c"
N_SESSION_COLOR = "gray"
N_TRIAL_COLOR = "purple"
NEW_DATASET_COLOR = "C0"  # solid blue in S1 old/new overlays
OLD_DATASET_COLOR = "C1"  # dashed orange in S1 old/new overlays
LEFT_COLOR = "salmon"
RIGHT_COLOR = "cornflowerblue"
MULTI_CHOICE_PALETTE = (LEFT_COLOR, RIGHT_COLOR, VIOLATION_COLOR)
CHOICE_ORDER = ("Left", "Violation", "Right")
CHOICE_PALETTE = (LEFT_COLOR, VIOLATION_COLOR, RIGHT_COLOR)
PREV_VIOLATION_COLOR = "#93c47d"
PREV_VIOLATION_FILTER_COLOR = "#38761d"
MODEL_COMPARE_COLOR = "darkblue"

PSYCHOMETRIC_ANIMALS = (
    "W060",
    "W066",
    "W068",
    "W073",
    "W074",
    "W075",
    "W080",
    "W081",
    "W083",
    "W088",
    "W089",
)
