"""Hand-built cases with known frontiers. Day 6's entire verdict rests on
this function, so it gets the same scrutiny as eval.py's metrics did on
Day 2."""

from retrieval.pareto import kendall_tau_b, pareto_frontier


def _cell(cid, recall5, p95_ms, status="ok"):
    return {
        "config_id": cid,
        "run": {"status": status},
        "quality": {"recall@5": recall5},
        "cost": {"p95_ms": p95_ms},
    }


def test_single_cell_is_its_own_frontier():
    assert pareto_frontier([_cell("a", 0.8, 100)]) == {"a"}


def test_strictly_dominated_cell_is_excluded():
    # b has both worse recall and worse latency than a -> dominated
    cells = [_cell("a", 0.9, 100), _cell("b", 0.8, 200)]
    assert pareto_frontier(cells) == {"a"}


def test_tradeoff_cells_are_both_on_frontier():
    # a is faster but lower recall; b is slower but higher recall -- neither dominates
    cells = [_cell("a", 0.7, 50), _cell("b", 0.9, 150)]
    assert pareto_frontier(cells) == {"a", "b"}


def test_exact_ties_are_both_on_frontier():
    cells = [_cell("a", 0.8, 100), _cell("b", 0.8, 100)]
    assert pareto_frontier(cells) == {"a", "b"}


def test_same_recall_lower_latency_dominates():
    cells = [_cell("a", 0.8, 100), _cell("b", 0.8, 200)]
    assert pareto_frontier(cells) == {"a"}


def test_same_latency_higher_recall_dominates():
    cells = [_cell("a", 0.9, 100), _cell("b", 0.8, 100)]
    assert pareto_frontier(cells) == {"a"}


def test_oom_cells_are_never_frontier_members():
    cells = [_cell("a", 0.9, 100), _cell("b", 0.0, 0, status="oom")]
    assert pareto_frontier(cells) == {"a"}


def test_three_way_chain_only_extremes_and_nondominated_middle_survive():
    cells = [
        _cell("fast_low", 0.5, 10),
        _cell("mid", 0.8, 50),
        _cell("slow_high", 0.95, 200),
        _cell("dominated", 0.7, 60),  # worse than mid on both axes
    ]
    assert pareto_frontier(cells) == {"fast_low", "mid", "slow_high"}


# --- kendall_tau_b -------------------------------------------------------------


def test_kendall_tau_perfectly_concordant_is_one():
    pairs = [(1, 10), (2, 20), (3, 30), (4, 40)]
    assert kendall_tau_b(pairs) == 1.0


def test_kendall_tau_perfectly_reversed_is_minus_one():
    pairs = [(1, 40), (2, 30), (3, 20), (4, 10)]
    assert kendall_tau_b(pairs) == -1.0


def test_kendall_tau_no_relationship_is_near_zero():
    # x increasing, y bouncing so concordant/discordant pairs roughly cancel
    pairs = [(1, 10), (2, 40), (3, 20), (4, 30)]
    tau = kendall_tau_b(pairs)
    assert -0.4 < tau < 0.4


def test_kendall_tau_single_pair_is_zero_no_comparisons_possible():
    assert kendall_tau_b([(1, 1)]) == 0.0


def test_kendall_tau_handles_ties_on_one_side():
    # x has a tie (2,2); still well-defined via tau-b's denominator correction
    pairs = [(1, 10), (2, 20), (2, 25), (3, 30)]
    tau = kendall_tau_b(pairs)
    assert 0.9 <= tau <= 1.0  # still strongly concordant
