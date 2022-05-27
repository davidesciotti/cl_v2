import numpy as np
# from scipy.interpolate import interp1d
# from classy import Class
# import matplotlib.pyplot as plt

# setting the ell bounds
ell_min_GC = 10
ell_max_GC = 5000
ell_min_WL = 10
ell_max_WL = 5000
zbin    = 10

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

#genero gli EDGES, sono ancora lineari 
ell_bins = 20

ell_GC = np.logspace(np.log10(ell_min_GC), np.log10(ell_max_GC), ell_bins+1)
ell_WL = np.logspace(np.log10(ell_min_WL), np.log10(ell_max_WL), ell_bins+1)

l_centr_GC   = np.zeros((ell_bins))
l_centr_WL   = np.zeros((ell_bins))
logaritmi_GC = np.zeros((ell_bins))
logaritmi_WL = np.zeros((ell_bins))
delta_l_GC   = np.zeros((ell_bins))
delta_l_WL   = np.zeros((ell_bins))
l_lin_GC     = np.zeros((ell_bins))
l_lin_WL     = np.zeros((ell_bins))

#prendo i valori centrali in scala lineare
for i in range(ell_bins):
  l_centr_GC[i] = (ell_GC[i+1]+ell_GC[i])/2
  l_centr_WL[i] = (ell_WL[i+1]+ell_WL[i])/2 

#genero i delta prima di passare in scala log
k=0
for i in range(ell_bins): 
  delta_l_GC[k] = ell_GC[ell_bins-i]-ell_GC[ell_bins-i-1]
  delta_l_WL[k] = ell_WL[ell_bins-i]-ell_WL[ell_bins-i-1]
  k = k+1
delta_l_GC = np.flip(delta_l_GC)
delta_l_WL = np.flip(delta_l_WL) 
# xxx occhio a questi flip. Sembra funzionare tutto ma in XC per uno dei due non lo facevo...

#prendo il log10 dei valori centrali 
for i in range(ell_bins):
    logaritmi_GC[i] = np.log10(l_centr_GC[i])
    logaritmi_WL[i] = np.log10(l_centr_WL[i])


ell_GC = logaritmi_GC #GC
ell_WL = logaritmi_WL #WL
ell_XC = ell_GC # anche XC va da 1 a 3000 

# xxx note: they'se all equal, for the time being
np.savetxt(r"%s\output\ell_values\ell_WL.dat" %path, ell_WL)
np.savetxt(r"%s\output\ell_values\ell_XC.dat" %path, ell_XC)
np.savetxt(r"%s\output\ell_values\ell_GC.dat" %path, ell_GC)



