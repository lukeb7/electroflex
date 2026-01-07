from scripts.promote_if_better import decide_promotion


def test_promotes_when_no_champion():
    d = decide_promotion(candidate_metric=0.5, champion_metric=None, improvement_threshold=0.01)
    assert d.promote is True
    assert d.reason == "no_champion"


def test_promotes_when_improved_enough():
    # champion 1.0, threshold 1% => required <= 0.99
    d = decide_promotion(candidate_metric=0.99, champion_metric=1.0, improvement_threshold=0.01)
    assert d.promote is True


def test_does_not_promote_when_not_improved_enough():
    d = decide_promotion(candidate_metric=0.995, champion_metric=1.0, improvement_threshold=0.01)
    assert d.promote is False