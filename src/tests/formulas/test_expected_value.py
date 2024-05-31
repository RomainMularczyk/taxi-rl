from lib.formulas.q_learning import QLearning


def test_six_face_dice_expected_value():
    result = QLearning.expected_value(6, 1/6)
    expected = 3.5
    assert result == expected


def test_zero_expected_value():
    result = QLearning.expected_value(0, 1/10)
    expected = 0.0
    assert result == expected
