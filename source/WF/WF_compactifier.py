import numpy as np
from scipy.integrate import quad, dblquad, nquad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from numba import jit
import os
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
start = time.time()


path = "C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/IA_sylvain_25feb"
zbins = 10

# z = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/windows_sylvain/nz_source/z.txt")
# n_rows = z.shape[0]

# W_IA = np.zeros((n_rows, zbins +1))
# W_IA_Fofz = np.zeros((n_rows, zbins +1))

# W_IA[:,0] = z
# W_IA_Fofz[:,0] = z

# for i in range(zbins):
#     W_IA[:,i+1] = np.genfromtxt("%s/W_IA_%i.txt" %(path, i+1))
#     W_IA_Fofz[:,i+1] = np.genfromtxt("%s/W_IA_Fofz%i.txt" %(path, i+1))
    
# np.savetxt("%s/W_IA_Fofz.txt" %(path), W_IA_Fofz)

# building the noIA WiL 
# W_WL_pyssc_convention = np.genfromtxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/everyones_WFs/sylvain/W_WL_pyssc_convention.txt")

# W_WL_pyssc_convention_IAtest = np.zeros((n_rows, zbins +1)) # create array
# W_WL_pyssc_convention_IAtest[1:,:] = W_WL_pyssc_convention[1:,:] - W_IA_Fofz[1:,:] #subtract the IA term
# W_WL_pyssc_convention_IAtest[:,0] = z

# # save
# np.savetxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/everyones_WFs/sylvain/IA/W_WL_pyssc_convention_IAtest.txt", W_WL_pyssc_convention_IAtest)


# np.savetxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/everyones_WFs/sylvain/IA/W_IA.txt", W_IA)
# np.savetxt("C:/Users/dscio/Documents/Lavoro/Programmi/SSC_comparison/data/everyones_WFs/sylvain/IA/W_IA_Fofz.txt", W_IA_Fofz)


# cheching
# W_IA_Fofz_10_test = np.genfromtxt("%s/W_IA_Fofz10.txt" %(path))
# plt.plot(W_IA_Fofz[:,0], W_IA_Fofz[:,10], label = "W_IA")
# plt.plot(z, W_IA_Fofz_10_test, "--", label = "W_IA_Fofz_10_test")
# ok, it works



# just a check
array = np.zeros((3,3))

for i in range(3):
    for j in range(3):
        if j>=i:
            array[i,j] = 1

print(array)

for i in range(3):
    for j in range(3):
        if j<i:
            array[i,j] = array[j,i]

print(array)