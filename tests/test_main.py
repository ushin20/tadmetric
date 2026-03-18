import pytest
import numpy as np
import tadmetric as tm


def test_package_exports_tadmetric():
    assert tm.Tadmetric is not None


def test_point_wise_evaluation_returns_expected_counts():
    metric = tm.Tadmetric(
        score=[0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
        label=[0, 0, 1, 1, 1, 0, 0],
    )

    result = metric.evaluate(0.5, mode="point-wise")

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(0.8)
    assert (result.tp, result.tn, result.fp, result.fn) == (2, 4, 0, 1)


def test_point_adjusted_fills_detected_event_and_reports_latency():
    metric = tm.Tadmetric(
        score=[0.1, 0.2, 0.4, 0.9, 0.2, 0.1, 0.0],
        label=[0, 0, 1, 1, 1, 0, 0],
    )

    result = metric.evaluate(0.5, mode="point-adjusted", calc_latency=True)

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.latency == pytest.approx(1.0)


def test_composite_uses_point_precision_and_event_recall():
    metric = tm.Tadmetric(
        score=[0.1, 0.2, 0.4, 0.9, 0.2, 0.1, 0.0],
        label=[0, 0, 1, 1, 1, 0, 0],
    )

    result = metric.evaluate(0.5, mode="composite")

    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.detected_events == 1
    assert result.total_events == 1


def test_search_without_range_uses_score_min_max_grid():
    metric = tm.Tadmetric(
        score=[0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
        label=[0, 0, 1, 1, 1, 0, 0],
    )

    result = metric.search(mode="point-adjusted")

    assert result.f1 == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.9)
    assert result.candidates_evaluated == 100


def test_search_with_range_uses_grid_thresholds():
    metric = tm.Tadmetric(
        score=[0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
        label=[0, 0, 1, 1, 1, 0, 0],
    )

    result = metric.search(start=0.0, end=1.0, steps=11, verbose=False, mode="point-wise")

    assert result.threshold == pytest.approx(0.4)
    assert result.f1 == pytest.approx(1.0)
    assert result.candidates_evaluated == 11


def test_update_validates_lengths():
    metric = tm.Tadmetric(score=[0.1, 0.2], label=[0, 1])

    with pytest.raises(ValueError, match="same length"):
        metric.update(score=[0.1, 0.2], label=[0])


def test_search_verbose_progress_bar_on_large_input():
    len = 1000000
    score = np.linspace(0.0, 1.0, num=len)
    label = np.zeros(len, dtype=int)
    label[4_000:6_000] = 1

    metric = tm.Tadmetric(score=score, label=label)
    result = metric.search(mode="point-wise", steps=100, verbose=True)
    
    
if __name__ == "__main__":
    test_search_verbose_progress_bar_on_large_input()