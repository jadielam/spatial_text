from spatial_text.utils.running_stats import RunningStats


def test_mean():
    rs = RunningStats()
    rs.push(1)
    rs.push(2)
    rs.push(3)
    assert rs.mean() == 2.0


def test_std():
    rs = RunningStats()
    rs.push(1)
    rs.push(2)
    rs.push(3)
    assert rs.standard_deviation() == 1.0


def test_zero_mean():
    rs = RunningStats()
    assert rs.mean() == 0.0


def test_zero_std():
    rs = RunningStats()
    assert rs.standard_deviation() == 0.0
