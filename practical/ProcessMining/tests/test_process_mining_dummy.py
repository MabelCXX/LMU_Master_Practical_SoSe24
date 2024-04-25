import numpy as np

from practical.ProcessMining.process_mining_dummy import k_means


def test_k_means():
    np.random.seed(0)
    test_data = np.random.randn(100, 2)

    labels, centroids = k_means(test_data, k=3)

    assert labels.shape == (100,), "Labels incorrect."
    assert centroids.shape == (3, 2), "Centroids incorrect."

    assert not np.any(np.isnan(labels)), "Some data points not assigned to any cluster."

    for i in range(3):
        points_in_cluster = test_data[labels == i]
        cluster_mean = np.mean(points_in_cluster, axis=0)
        assert np.allclose(cluster_mean, centroids[i]), f"Centroid {i} is not the mean of its cluster."

    print("All tests passed!")


test_k_means()
