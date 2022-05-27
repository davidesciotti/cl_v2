import numpy as np
#import scipy.integrate as integrate
from scipy.interpolate import interp1d
#from classy import Class
import matplotlib.pyplot as plt

def somma(zbin):
    risultato = 0
    for i in range(zbin):
        risultato += zbin - i
    return risultato

# uploading the index array

ind = np.genfromtxt(r"C:\Users\dscio\Documents\Lavoro\Programmi\indici.dat")
ind = ind.astype(int)
ind = ind - 1 #xxx attenzione!!!

# setting the ell bounds
ell_min_GC = 10
ell_max_GC = 5000
ell_min_WL = 10
ell_max_WL = 5000
zbin    = 10

# symmetric and asymmetric indexes
ind_simmetrici  = somma(zbin) # this is 55
ind_Asimmetrici = zbin**2     # this is 100
ind_tot         = 2 * ind_simmetrici + ind_Asimmetrici # this is 210

path = r"C:\Users\dscio\Documents\Lavoro\Programmi\Cij_davide"

##########################################################

#####################################################

C_LL = np.zeros((ell_bins, zbin, zbin))
C_LG = np.zeros((ell_bins, zbin, zbin))
C_GG = np.zeros((ell_bins, zbin, zbin))

#      C_LL_WL = np.zeros((ell_bins, zbin, zbin))

C_LL_2x2 = np.zeros((ell_bins, ind_simmetrici))
C_LG_2x2 = np.zeros((ell_bins, ind_Asimmetrici))
C_GG_2x2 = np.zeros((ell_bins, ind_simmetrici))

#      C_LL_2x2_WL = np.zeros((ell_bins, ind_simmetrici))


C_LL_full = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijLL-LCDM-NonLin-eNLA.dat" %path)
C_LG_full = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijLG-LCDM-NonLin-eNLA.dat" %path)
C_GG_full = np.genfromtxt(r"%s\data\Cij-NonLin-eNLA_15gen\CijGG-LCDM-NonLin-eNLA.dat" %path)

# switching to log scale
C_LL_full[:,0] = np.log10(C_LL_full[:,0])
C_LG_full[:,0] = np.log10(C_LG_full[:,0])
C_GG_full[:,0] = np.log10(C_GG_full[:,0])

for j in range(ind_simmetrici):
  fLL = interp1d(C_LL_full[:,0], C_LL_full[:,j+1], kind='linear')
  fGG = interp1d(C_GG_full[:,0], C_GG_full[:,j+1], kind='linear')
  C_LL_2x2[:,j] = fLL(ell_WL) 
  C_GG_2x2[:,j] = fGG(ell_GC) 
  
#        C_LL_2x2_WL[:,j] = fLL(ell_WL)

for j in range(ind_Asimmetrici):
  fLG = interp1d(C_LG_full[:,0], C_LG_full[:,j+1], kind='linear')
  C_LG_2x2[:,j] = fLG(ell_XC) 
  

# saving these to compare them with Sylvain's
np.savetxt(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\CijLL2x2.dat" %path, C_LL_2x2)
np.savetxt(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\CijLG2x2.dat" %path, C_LG_2x2)
np.savetxt(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\CijGG2x2.dat" %path, C_GG_2x2)

##############################################
#riempio le matrici zbinxzbin
h = zbin
for m in range(ell_bins): 
    for i in range(zbin):     
        for j in range(zbin):
            C_LG[m,i,j] = C_LG_2x2[m, j+i*h]

for m in range(ell_bins): 
    k=0
    for i in range(zbin):     
        for j in range(zbin):
          if j>=i:                  
            C_LL[m,i,j] = C_LL_2x2[m, k]
            C_GG[m,i,j] = C_GG_2x2[m, k] 
            k = k+1
            
#      for m in range(ell_bins): 
#        k=0
#        for i in range(zbin):     
#          for j in range(zbin):  
#            if j>=i:
#              C_LL_WL[m,i,j] = C_LL_2x2_WL[m, k]
#              k=k+1

#simmetrizzo
for m in range(ell_bins): 
    for i in range(zbin):     
        for j in range(zbin):
          C_LL[m,j,i] = C_LL[m,i,j]
          C_GG[m,j,i] = C_GG[m,i,j]

#      for m in range(ell_bins): 
#        k=0
#        for i in range(zbin):     
#          for j in range(zbin):           
#            C_LL_WL[m,j,i] = C_LL_WL[m,i,j]
#      

# salvo le matrici:
np.save(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_LL_nbl%i.npy" %(path, ell_bins), C_LL)
np.save(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_LG_nbl%i.npy" %(path, ell_bins), C_LG)
np.save(r"%s\data\Cij-NonLin-eNLA_15gen\Cij_reshape\C_GG_nbl%i.npy" %(path, ell_bins), C_GG)

# just to check if the interpolation works
# plt.plot(C_LL_full[:,0], C_LL_full[:,1], "b", label = "C_LL_full")
# plt.plot(ell_WL, C_LL_2x2[:,0], "r.", label = "C_LL_2x2")
plt.plot(ell_WL, C_LL[:,0,0], "g.-", label = "C_LL")
plt.legend()
########################################################
 
# # riempio il datavector per WL + XC + GCph
# D_all = np.zeros((ell_bins, 2, 2, zbin, zbin))

# for elle in range(ell_bins):
#   for i in range(zbin):
#     for j in range(zbin):
#       D_all[elle,0,1,i,j] = C_LG[elle,j,i]
#       D_all[elle,1,0,i,j] = C_LG[elle,i,j] 
      
# for elle in range(ell_bins):
#   for i in range(zbin):
#     for j in range(i+1):
#       D_all[elle,0,0,i,j] = C_LL[elle,i,j]
#       D_all[elle,1,1,i,j] = C_GG[elle,i,j] 
      
# #simmetrizzo
# for elle in range(ell_bins):
#   for j in range(zbin):
#     for i in range(zbin):
#       D_all[elle,0,0,j,i] = D_all[elle,0,0,i,j] 
#       D_all[elle,1,1,j,i] = D_all[elle,1,1,i,j] 

# np.save(r"%s\output\matrici_base\D_all_nbl%i.npy" %(path, ell_bins), D_all)

# # riempio il datavector per WL  + GCph
# D_reduced = np.zeros((ell_bins, 2, zbin, zbin))
      
# for elle in range(ell_bins):
#   for i in range(zbin):
#     for j in range(i+1):
#       D_reduced[elle,0,i,j] = C_LL[elle,i,j]
#       D_reduced[elle,1,i,j] = C_GG[elle,i,j] 
      
# #simmetrizzo
# for elle in range(ell_bins):
#   for j in range(zbin):
#     for i in range(zbin):
#       D_reduced[elle,0,j,i] = D_reduced[elle,0,i,j] 
#       D_reduced[elle,1,j,i] = D_reduced[elle,1,i,j] 

# np.save(r"%s\output\matrici_base\D_reduced_nbl%i.npy" %(path, ell_bins), D_reduced)
