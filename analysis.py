import mysymnmf
import kmeans
import sklearn.metrics as sk
import sys
import pandas as pd
import numpy as np
import math

# calculate the normalized similarity matrix using interface
def norm(points, n_points, dim):
    output = mysymnmf.norm(points, n_points, dim)
    return output

# calculate relavent output matrix with symNMF using interface
def symnmf(k, points, n_points, dim):
    np.random.seed(0)
    W = norm(points, n_points, dim)
    m = np.mean(W)
    H = np.random.uniform(0, 2 * math.sqrt(m/k), size=(n_points, k))
    H_list = H.tolist()
    output = mysymnmf.symnmf(H_list, W, n_points, k)
    return output


def main():
    # verify correct amount of arguments
    try:

        k = sys.argv[1]
        file_name = sys.argv[2]

        # to read file we will use try-except block as learned
        data = pd.read_csv(file_name, header=None)

        points = [x.tolist() for index, x in data.iterrows()]
        num_points = len(points)
        if (int(k) >= num_points) or (num_points == 0):
            print("An Error Has Occurred")
            sys.exit(1)

        # calculate silhouette score
        kmeanslist = kmeans.kmeans(points, int(k))
        H = symnmf(int(k), points, num_points, len(points[0]))
        symnmflist = mysymnmf.analysis(H, num_points, int(k))
        nmf = sk.silhouette_score(points, symnmflist)
        kmean = sk.silhouette_score(points, kmeanslist)

    except Exception as e:
        print("An Error Has Occurred")
        # sys.exit(1)
        raise(e)

    print("nmf: {:.4f}".format(nmf))
    print("kmeans: {:.4f}".format(kmean))


if __name__ == "__main__":
    main()


