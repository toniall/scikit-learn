import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster._k_means_elkan import k_means_elkan
from sklearn.cluster.k_means_ import k_means, _k_init

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal


def test_elkan_results():
    rnd = np.random.RandomState(0)
    X_normal = rnd.normal(size=(50, 10))
    X_blobs, _ = make_blobs(random_state=0)
    for X in [X_normal, X_blobs]:
        X -= X.mean(axis=0)
        init = _k_init(X, n_clusters=5, random_state=1)
        loyd_means, loyd_labels, _ = k_means(X, n_clusters=5, init=init,
                                             n_init=1, verbose=10)
        elkan_means, elkan_labels = k_means_elkan(X, n_clusters=5, init=init,
                                                  verbose=True)
        assert_array_almost_equal(loyd_means, elkan_means)
        assert_array_equal(loyd_labels, elkan_labels)
