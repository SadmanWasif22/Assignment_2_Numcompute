import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats


def test_streaming_stats_mean_single_chunk():
    stats = StreamingStats()

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    stats.update_stats(X)

    assert np.allclose(stats.mean(), np.array([2.0, 3.0]))


def test_streaming_stats_variance_single_chunk():
    stats = StreamingStats()

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    stats.update_stats(X)

    assert np.allclose(stats.variance(), np.array([1.0, 1.0]))


def test_streaming_stats_partial_updates():
    stats = StreamingStats()

    stats.update_stats(np.array([[1.0, 2.0]]))
    stats.update_stats(np.array([[3.0, 4.0]]))

    assert np.allclose(stats.mean(), np.array([2.0, 3.0]))
    assert np.allclose(stats.variance(), np.array([1.0, 1.0]))


def test_streaming_stats_nan_handling():
    stats = StreamingStats()

    X = np.array([
        [1.0, np.nan],
        [3.0, 4.0],
    ])

    stats.update_stats(X)

    assert np.allclose(stats.mean(), np.array([2.0, 4.0]))
    assert not np.any(np.isnan(stats.mean()))


def test_streaming_stats_min_max():
    stats = StreamingStats()

    X = np.array([
        [5.0, 2.0],
        [1.0, 9.0],
        [3.0, 4.0],
    ])

    stats.update_stats(X)

    assert np.allclose(stats.min(), np.array([1.0, 2.0]))
    assert np.allclose(stats.max(), np.array([5.0, 9.0]))


def test_streaming_stats_std_shape():
    stats = StreamingStats()

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    stats.update_stats(X)

    assert stats.std().shape == (2,)


def test_streaming_stats_quantile():
    stats = StreamingStats(store_values=True)

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    stats.update_stats(X)

    q = stats.quantile(0.5)

    assert np.allclose(q, np.array([3.0, 4.0]))


def test_streaming_stats_histogram():
    stats = StreamingStats(store_values=True, histogram_bins=2)

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    stats.update_stats(X)

    hist = stats.histogram()

    assert 0 in hist
    assert 1 in hist
    assert len(hist[0][0]) == 2


def test_streaming_stats_result_keys():
    stats = StreamingStats()

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    stats.update_stats(X)
    result = stats.result()

    assert "count" in result
    assert "mean" in result
    assert "variance" in result
    assert "std" in result
    assert "min" in result
    assert "max" in result


def test_streaming_stats_invalid_shape():
    stats = StreamingStats()

    with pytest.raises(ValueError):
        stats.update_stats(np.array([1.0, 2.0, 3.0]))


def test_streaming_stats_empty_array():
    stats = StreamingStats()

    with pytest.raises(ValueError):
        stats.update_stats(np.empty((0, 2)))


def test_streaming_stats_result_before_update():
    stats = StreamingStats()

    with pytest.raises(RuntimeError):
        stats.result()