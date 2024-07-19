import math
import sys
import pandas as pd
import numpy as np
import random
import mysymnmf

np.random.seed(0)

#for each function needed, calling the releavent function with the interface and return the result
def sym(points, n_points, dim):
    result = mysymnmf.sym(points, n_points, dim)
    return result


def ddg(points, n_points, dim):
    result = mysymnmf.ddg(points, n_points, dim)
    return result


def norm(points, n_points, dim):
    result = mysymnmf.norm(points, n_points, dim)
    return result


def symnmf(k, points, n_points, dim):
    W = norm(points, n_points, dim)
    m=np.mean(W)
    H=np.random.uniform(0,2*math.sqrt(m/k),size=(n_points,k))
    H_list = H.tolist()
    result = mysymnmf.symnmf(H_list, W, n_points, k)
    return result


def main():
    #check right anount of args
    if (len(sys.argv) < 3 or len(sys.argv) > 4):
        print("An Error Has Occurred")
        sys.exit(1)

    k = sys.argv[1]
    goal = sys.argv[2]
    file_name = sys.argv[3]
    
    try: 
        #read file
        data = pd.read_csv(file_name, header=None)
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

    points = [x.tolist() for index, x in data.iterrows()]
    if int(k) >= len(points) or len(points) == 0:
        print("An Error Has Occurred")
        sys.exit(1)

    #call of the required function
    try: 
        if goal == "sym":
            res = sym(points, len(points) , len(points[0]))
        elif goal == "ddg":
            res = ddg(points, len(points) , len(points[0]))
        elif goal == "norm":
            res = norm(points, len(points) , len(points[0]))
        elif goal == "symnmf":
            res = symnmf(int(k), points, len(points) , len(points[0]))
        else:
            print("An Error Has Occurred")
            sys.exit(1)
    except Exception as e:
        print("An Error Has Occurred")
        sys.exit(1)

    #print the result matrix
    for row in res:
        print(",".join(str("{:.4f}".format(round(x, 4))) for x in row))


if __name__ == "__main__":
    main()
