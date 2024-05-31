from lib.models.Action import Action


def test_available_actions():
    result = Action.available_actions({
        "prob": 1.0,
        "action_mask": [0, 0, 1, 1, 0, 1]  # type: ignore
    })
    expected = ("EAST", "WEST", "DROP_OFF")
    assert result == expected


def test_no_available_actions():
    result = Action.available_actions({
        "prob": 1.0,
        "action_mask": [0, 0, 0, 0, 0, 0]  # type: ignore
    })
    expected = ()
    assert result == expected


def test_all_available_actions():
    result = Action.available_actions({
        "prob": 1.0,
        "action_mask": [1, 1, 1, 1, 1, 1]  # type: ignore
    })
    expected = ("SOUTH", "NORTH", "EAST", "WEST", "PICK_UP", "DROP_OFF")
    assert result == expected
