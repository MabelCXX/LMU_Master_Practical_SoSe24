import numpy as np


def k_means(x, k, max_iters=100):
    centroids = x[np.random.choice(range(len(x)), k, replace=False)]
    labels = np.zeros(len(x))
    for _ in range(max_iters):
        old_labels = labels.copy()
        for i, x in enumerate(x):
            labels[i] = np.argmin([np.linalg.norm(x - centroid) ** 2 for centroid in centroids])
        for j in range(k):
            points = x[labels == j]
            if points.any():
                centroids[j] = np.mean(points, axis=0)
        if np.all(labels == old_labels):
            break
    return labels, centroids
