from spatial_text.text.edit_distance import edit_distance


def test_equal_strings():
    assert edit_distance('abc', 'abc') == 0


def test_insertion():
    assert edit_distance('abc', 'abcd') == 1


def test_deletion():
    assert edit_distance('abcd', 'abc') == 1


def test_substitution():
    assert edit_distance('abc', 'abd') == 1


def test_with_empty_string():
    assert edit_distance('', 'abc') == 3
    assert edit_distance('abc', '') == 3
