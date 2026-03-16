import tadmetric as tm


def test_evaluator_metric_returns_prfscore():
    e = tm.evaluator(
        [0, 0, 1, 1, 1, 0, 0],
        [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
    )

    pa = e.point_adjusted(thr=0.3)
    point = e.point_wise(thr=0.5)

    assert pa.f1 == 1.0
    assert point.precision == 1.0
    assert point.recall == 2 / 3
    assert tuple(point) == (point.precision, point.recall, point.f1)


def test_threshold_defaults_to_best_for_metric():
    e = tm.evaluator(
        [0, 0, 1, 1, 1, 0, 0],
        [0.1, 0.2, 0.4, 0.9, 0.2, 0.1, 0.0],
    )

    best = e.best(metric="point_adjusted")

    default_result = e.point_adjusted()

    assert default_result.precision == best.evaluation["point_adjusted"].precision
    assert default_result.f1 == best.f1


def test_point_adjusted_uses_default_k_100():
    e = tm.evaluator(
        [0, 0, 1, 1, 1, 0, 0],
        [0.1, 0.2, 0.8, 0.9, 0.2, 0.1, 0.0],
    )

    assert e.point_adjusted().f1 == e.point_adjusted(k=100).f1


def test_evaluator_caches_metric_threshold_results():
    e = tm.evaluator(
        [0, 0, 1, 1, 1, 0, 0],
        [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
    )

    first = e.point_adjusted(thr=0.3).f1
    cache_size_after_first = len(e._metric_cache)
    second = e.point_adjusted(thr=0.3).f1

    assert first == second
    assert len(e._metric_cache) == cache_size_after_first


def test_evaluator_reevaluate_resets_caches():
    e = tm.evaluator([0, 1, 1], [0.1, 0.8, 0.3])

    before = e.point_wise(thr=0.5).f1
    assert e._metric_cache

    e.reevaluate([0, 1, 1], [0.1, 0.8, 0.9])
    after = e.point_wise(thr=0.5).f1

    assert before != after


def test_evaluator_best_matches_functional_search():
    y_true = [0, 0, 1, 1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0]

    e = tm.evaluator(y_true, y_score)
    best_from_evaluator = e.best(metric="composite")
    best_from_function = tm.search_best_f1_threshold(y_true, y_score, metric="composite")

    assert best_from_evaluator.threshold == best_from_function.threshold
    assert best_from_evaluator.f1 == best_from_function.f1
    assert best_from_evaluator.evaluation.asdict() == best_from_function.evaluation.asdict()


def test_api_overview_mentions_direct_evaluator_usage():
    overview = tm.api_overview()

    assert "evaluator(y_true, y_score)" in overview
    assert "e.point_adjusted(" in overview


def test_unpacking_point_wise_result_feels_natural():
    e = tm.evaluator(
        [0, 0, 1, 1, 1, 0, 0],
        [0.1, 0.2, 0.4, 0.9, 0.7, 0.1, 0.0],
    )

    pre, rec, f1 = e.point_wise(thr=0.5)

    assert pre == 1.0
    assert rec == 2 / 3
    assert f1 == 0.8
