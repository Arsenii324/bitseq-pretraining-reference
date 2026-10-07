"""Independent target/brute/path references for the actual finite instrument."""

import itertools
import math

import numpy as np
import pytest
import torch


def parity_fixture(graph, swapped=False):
    """Order12 favors equal bits, order21 unequal; terminal law stays uniform."""
    table = np.full((len(graph.boards), 2, 2), -math.log(2))
    for n, board in enumerate(graph.boards):
        for i in range(2):
            if board[i] == -1 and board[1 - i] != -1:
                same = 0.9 if i == 1 else 0.1
                if swapped:
                    same = 1 - same
                table[n, i, board[1 - i]] = math.log(same)
                table[n, i, 1 - board[1 - i]] = math.log(1 - same)
    return table


def brute(graph, table, order):
    result, entropy, second = [], [], []
    for y in itertools.product(range(graph.vocab), repeat=graph.length):
        weights = []
        orders = (
            [tuple(range(graph.length))]
            if order == "ar"
            else list(itertools.permutations(range(graph.length)))
        )
        for sigma in orders:
            board = [-1] * graph.length
            logweight = -math.log(len(orders))
            for i in sigma:
                row = int(graph.index(np.array([board]))[0])
                logweight += table[row, i, y[i]]
                board[i] = y[i]
            weights.append(math.exp(logweight))
        total = math.fsum(weights)
        result.append(total)
        entropy.append(-math.fsum(w / total * math.log(w / total) for w in weights))
        second.append(math.fsum(w / total * math.log(w) ** 2 for w in weights))
    return np.array(result), np.array(entropy), np.array(second)


def test_graph_teacher_and_task_truth_are_independent_and_complete():
    from bitseq.oracle import FiniteGraph, task_distribution, teacher_table
    from bitseq.evaluate import evaluate_exact, denoising_excess

    graph = FiniteGraph()
    assert len(graph.boards) == 15625 and len(graph.src) == 75000
    np.testing.assert_array_equal(graph.index(graph.boards), np.arange(15625))
    assert np.bincount(task_distribution(graph, 12)["distance"]).tolist() == [
        8,
        96,
        528,
        1600,
        1672,
        192,
    ]
    for alpha, expected_r in [(12, 0.06315933141383001), (24, 0.28255342346036993)]:
        target = task_distribution(graph, alpha)
        logwords, masses = teacher_table(graph, target["q"])
        for order in ["ar", "random"]:
            result = evaluate_exact(graph, logwords, order)
            np.testing.assert_allclose(result["p"], target["q"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                result["conditional_entropy"],
                0 if order == "ar" else math.log(720),
                atol=1e-9,
            )
            np.testing.assert_allclose(result["layer_mass"], 1, atol=1e-10)
            assert abs(result["visits_total"] - 7) < 1e-10
        assert abs(np.dot(target["q"], target["reward"]) - expected_r) < 1e-12
        excess = denoising_excess(graph, logwords, target["q"])
        assert abs(excess["excess_kl"]) < 1e-12
        assert abs(excess["cross_entropy_minus_teacher_entropy"]) < 1e-10


@pytest.mark.parametrize("order", ["ar", "random"])
def test_inconsistent_table_moments_match_every_path(order):
    from bitseq.oracle import FiniteGraph
    from bitseq.evaluate import evaluate_exact

    graph = FiniteGraph(3, 2)
    table = np.zeros((27, 3, 2))
    for n, s in enumerate(graph.boards):
        for i in range(3):
            p = (
                0.13
                + 0.71
                * ((sum((j + 1) * (v + 2) for j, v in enumerate(s)) + 3 * i) % 11)
                / 10
            )
            table[n, i] = np.log([p, 1 - p])
    result = evaluate_exact(graph, table, order)
    p, h, m2 = brute(graph, table, order)
    np.testing.assert_allclose(result["p"], p, atol=1e-12)
    np.testing.assert_allclose(result["conditional_entropy"], h, atol=1e-11)
    np.testing.assert_allclose(result["logpath_second"], m2, atol=1e-10)
    assert (
        abs(result["joint_entropy"] - result["terminal_entropy"] - np.dot(p, h)) < 1e-11
    )


def test_entropy_is_not_reference_preservation_and_clone_is_zero():
    from bitseq.oracle import FiniteGraph
    from bitseq.evaluate import evaluate_exact

    graph = FiniteGraph(2, 2)
    ref = parity_fixture(graph)
    pol = parity_fixture(graph, True)
    result = evaluate_exact(graph, pol, "random", reference=ref)
    np.testing.assert_allclose(result["p"], 0.25, atol=1e-12)
    np.testing.assert_allclose(
        result["conditional_entropy"], 0.3250829733914482, atol=1e-12
    )
    assert abs(result["path_kl_ref_to_current"] - 0.8 * math.log(9)) < 1e-12
    assert abs(result["path_kl_current_to_ref"] - 0.8 * math.log(9)) < 1e-12
    clone = evaluate_exact(graph, ref, "random", reference=ref)
    assert abs(clone["path_kl_ref_to_current"]) < 1e-12
    assert abs(clone["path_kl_current_to_ref"]) < 1e-12
    assert (
        abs(result["fixed_reference_conditional_entropy"] - 0.3250829733914482) < 1e-12
    )


def test_logspace_retains_rare_terminal_conditional_entropy_without_end_renormalization():
    from bitseq.oracle import FiniteGraph
    from bitseq.evaluate import evaluate_exact

    graph = FiniteGraph(2, 2)
    logits = np.tile([0.0, -1000.0], (9, 2, 1))
    result = evaluate_exact(
        graph, logits, "random"
    )  # already log-normalized within precision
    assert result["underflow_terminals"] > 0
    assert np.isfinite(result["logp"]).all()
    np.testing.assert_allclose(result["conditional_entropy"], math.log(2), atol=1e-9)
    assert abs(result["p"].sum() - 1) < 1e-12
    with pytest.raises(ValueError, match="normaliz"):
        evaluate_exact(graph, logits + 0.1, "random")


def test_model_cache_and_native_enumeration_check_actual_probability_law():
    from bitseq.model import make_denoiser
    from bitseq.oracle import FiniteGraph, enumerated_env
    from bitseq.evaluate import cache_logwords, evaluate_exact, native_terminal_check

    graph = FiniteGraph()
    model = make_denoiser(107)
    table = cache_logwords(model, graph.boards, 256)
    assert table.dtype == np.float64 and table.shape == (15625, 6, 4)
    env = enumerated_env(graph)
    assert env.is_enumerable
    result = evaluate_exact(graph, table, "random")
    pmf = native_terminal_check(graph, table, "random")
    np.testing.assert_allclose(pmf, result["p"], rtol=2e-5, atol=1e-7)
    assert model.training  # cache restores mode
