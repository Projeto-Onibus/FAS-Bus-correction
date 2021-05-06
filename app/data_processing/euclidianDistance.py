import numpy as np
import cupy as cp

def Euclidian(traj1, traj2):
    if traj1.shape[0] < traj2.shape[0]:
        traj1,traj2 = (traj2,traj1)
    

def GetDiagonals(matrix):
    resultMatrix = np.zeros(matrix.shape)
    for offset in range(matrix.shape[0]):
        resultMatrix