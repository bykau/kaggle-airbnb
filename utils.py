import math
import numpy as np


def ndcg5(pred, actu):
    for i in range(len(pred)):
        idx = np.argsort(pred[i])[::-1].tolist().index(actu[i]) + 1
        if idx <= 5:
            sum += 1/math.log(1+idx, 2)
    return sum/len(pred)