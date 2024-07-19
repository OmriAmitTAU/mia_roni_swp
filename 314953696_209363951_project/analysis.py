import mysymnmf
import kmeans
import sklearn.metrics as sk
import sys
import pandas as pd
import numpy as np
import math

#calculate the norm matrix using the interface
def norm(points, n_points, dim):
    result = mysymnmf.norm(points, n_points, dim)
    return result

#calculate the result matrix with symnmf using the interface
def symnmf(k, points, n_points, dim):
    np.random.seed(0)
    W = norm(points, n_points, dim)
    m=np.mean(W)
    H=np.random.uniform(0,2*math.sqrt(m/k),size=(n_points,k))
    H_list = H.tolist()
    result = mysymnmf.symnmf(H_list, W, n_points, k)
    return result


def main():
    #check right amount of arguments
    if (len(sys.argv) < 2 or len(sys.argv) > 3):
        print("An Error Has Occurred")
        return

    k = sys.argv[1]
    file_name = sys.argv[2]

    try: 
        #read the file
        data = pd.read_csv(file_name, header=None)
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

    points = [x.tolist() for index, x in data.iterrows()]
    if int(k) >= len(points) or len(points) == 0:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        #calculation of silhouette score
        kmeanslist = kmeans.kmeans(points, int(k))
        H = symnmf(int(k), points, len(points), len(points[0]))
        symnmflist = mysymnmf.analysis(H, len(points), int(k))
        nmf = sk.silhouette_score(points, symnmflist)
        kmean = sk.silhouette_score(points, kmeanslist)
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

    print("nmf: {:.4f}".format(nmf))
    print("kmeans: {:.4f}".format(kmean))




if __name__ == "__main__":
    main()


