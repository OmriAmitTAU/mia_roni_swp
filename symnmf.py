import math
import sys
import pandas as pd
import numpy as np
import mysymnmf

np.random.seed(0)

# for each value of goal input, call relevant method with interface to return correct output
def sym(points, n_points, dim):
    output = mysymnmf.sym(points, n_points, dim)
    return output


def ddg(points, n_points, dim):
    output = mysymnmf.ddg(points, n_points, dim)
    return output


def norm(points, n_points, dim):
    output = mysymnmf.norm(points, n_points, dim)
    return output


def symnmf(k, points, n_points, dim):
    W = norm(points, n_points, dim)
    m = np.mean(W)
    H = np.random.uniform(0, 2 * math.sqrt(m / k), size=(n_points, k))
    H_list = H.tolist()
    output = mysymnmf.symnmf(H_list, W, n_points, k)
    return output


def main():
    try:
        k = sys.argv[1]
        goal = sys.argv[2]
        file_name = sys.argv[3]

        # to read file we will use try-except block as learned
        data = pd.read_csv(file_name, header=None)

        points = [x.tolist() for index, x in data.iterrows()]
        if int(k) >= len(points) or len(points) == 0:
            print("An Error Has Occurred")
            sys.exit(1)

        # call the required method
        if goal == "sym":
            mat = sym(points, len(points), len(points[0]))
        elif goal == "ddg":
            mat = ddg(points, len(points), len(points[0]))
        elif goal == "norm":
            mat = norm(points, len(points), len(points[0]))
        elif goal == "symnmf":
            mat = symnmf(int(k), points, len(points), len(points[0]))
        else:
            raise Exception


        # print the relevant output matrix
        for row in mat:
            print(",".join(str("{:.4f}".format(round(x, 4))) for x in row))
    
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
