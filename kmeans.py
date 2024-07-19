import numpy as np

# function to calculate Euclidean distance given two points
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# function to compute centroids based on first assignment
def compute_centroids(data, labels, k):
    centroids = []
    for i in range(k):
        cluster_points = [data[j] for j in range(len(data)) if labels[j] == i]
        if len(cluster_points) > 0:             # cluster is not empty
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = np.random.rand(len(data[0]))  # if cluster is empty apply random initialization
        centroids.append(centroid.tolist())
    return centroids

# function that assigns cluster labels to points based on first assignment
def assign_labels(data, centroids):
    labels = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        label = np.argmin(distances)
        labels.append(label)
    return labels

# kmeans function based on first assignment
def kmeans(data, k, max_iter=300, epsilon=0.0001):
    data = np.array(data)
    n, d = data.shape

    # initialize centroids randomly
    initial_indices = np.arange(k)
    centroids = data[initial_indices]

    for i in range(max_iter):
        # assign data points to clusters
        labels = assign_labels(data, centroids)

        # compute new centroids
        new_centroids = compute_centroids(data.tolist(), labels, k)

        # check for convergence
        if np.all(np.abs(np.array(new_centroids) - np.array(centroids)) < epsilon):
            break

        centroids = new_centroids

    return labels
