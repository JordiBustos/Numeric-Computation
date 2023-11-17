import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def generate_clusters(k):
    np.random.seed(0)
    data = []
    labels = []

    for i in range(k):
        cluster = np.random.normal(loc=i * 5, scale=2, size=(100, 3))
        data.append(cluster)
        labels.append(np.full(100, 0))

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels


def plot_points(data, labels, centroids=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=labels,
        cmap="viridis",
        s=30,
        label="Data Points",
    )

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c="red",
            marker="X",
            s=100,
            label="Centroids",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


def initialize_centroids(k, data):
    return np.random.uniform(low=data.min(), high=data.max(), size=(k, data.shape[1]))


def k_means_clustering(data, k, max_iter=1000, tolerance=1e-4):
    '''
      Implementation of the k-means clustering algorithm.
      Based on the pseudocode from the Wikipedia article:
      https://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm_(naive_k-means)
    '''
    centroids = initialize_centroids(k, data)

    for _ in range(max_iter):
        S = {i: [] for i in range(k)}

        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            S[np.argmin(distances)].append(point)

        new_centroids = []
        for i, cluster in S.items():
            if len(cluster) > 0:
                new_centroids.append(np.mean(cluster, axis=0))
            else:
                # If a cluster is empty, initialize its centroid randomly
                new_centroids.append(
                    np.random.uniform(
                        low=data.min(), high=data.max(), size=(data.shape[1])
                    )
                )

        new_centroids = np.array(new_centroids)

        if np.sum(np.abs(new_centroids - centroids)) < tolerance:
            break

        centroids = new_centroids

    labels = np.zeros(data.shape[0])
    for i, cluster in S.items():
        for point in cluster:
            labels[np.where(data == point)[0][0]] = i

    return centroids, labels


def main():
    k = int(input("Enter the number of clusters (k): "))
    data, labels = generate_clusters(k)
    plot_points(data, labels)

    centroids, labels = k_means_clustering(data, k=k)
    extended_data = np.concatenate([data, centroids])
    extended_labels = np.concatenate([labels, np.arange(k, k + len(centroids))])
    plot_points(extended_data, extended_labels)


if __name__ == "__main__":
    main()
