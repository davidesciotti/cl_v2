import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

zbins = 10
niz_davide = np.genfromtxt(r"%s\output\WFs_v3_cut\niz_e-20cut" %(path))
niz_vincenzo = np.genfromtxt(r"%s\data\dati_vincenzo\niNorm.dat" %path)

array = niz_vincenzo
minima_left, minima_right = [], []
redshift_left, redshift_right = [], []
for j in range(1,zbins+1):
    minima = []
    redshift = []
    for i in range(array.shape[0]):
        if array[i,j] < 1e-15 and array[i,j] != 0:
            minima.append(array[i,j])
            redshift.append(array[i,0])
            
    minima_left.append(minima[0]) 
    minima_right.append(minima[-1]) 
    redshift_left.append(redshift[0]) 
    redshift_right.append(redshift[-1]) 

redshift_left[0] = 0
redshift_right[-1] = 4
# minima = np.array((6e-19, 1e-22, 1e-19, 9e-20, 4e-20, 2e-20, 7e-21, 2e-21, 2e-22, 1e-17))
# minima_left = [6e-19, 3e-19, 1e-19, 9e-20, 4e-20, 2e-20, 7e-21, 2e-21, 2e-22, 1e-17]
# minima_right = [6e-19, 3e-19, 1e-19, 9e-20, 4e-20, 2e-20, 7e-21, 2e-21, 2e-22, 1e-17]
# minima = np.asarray(minima)
# plot(niz_davide, niz_vincenzo, niz_sylvain, column)

array = niz_davide
for i in range(array.shape[0]):
    for j in range(zbins):
        if (array[i,0] < redshift_left[j]):
            array[i,j+1] = 0
        if (array[i,0] > redshift_right[j]):
            array[i,j+1] = 0            
niz_davide = array
# plot(niz_davide, niz_vincenzo, niz_sylvain, column)


# np.savetxt(r"%s\output\WFs_v6_zcut\niz.txt" %path, niz_davide)

    