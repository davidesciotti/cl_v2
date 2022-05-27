from classy import Class
import numpy as np
import matplotlib.pyplot as plt

path = "/home/davide/Scrivania/PySSC-mio/programmi/Cij_davide"

#XXX v2: no z and k column and row in array "tabella"



#Start by specifying the cosmology
#xxx sto utilizzando i valori della Table 1 paper IST
#xxx voglio le omega grandi (Omega, non omega)
Omega_b = 0.05
Omega_m = 0.32
m_ncdm = 0.06
h = 0.67
Omega_ni = m_ncdm/(93.14*h*h)
Omega_cdm = Omega_m - Omega_b - Omega_ni
sigma8 = 0.816
n_s = 0.96
N_ncdm = 1
N_ur = 2.03351
k_max =50/h # xxx unità di misura? * o / h? kmax??

Omega_Lambda = 0.68
w0_fld = -1
wa_fld = 0.

#z_array = [0.2095, 0.489 , 0.619 , 0.7335, 0.8445, 0.9595, 1.087 , 1.2395, 1.45  , 2.038]
z_array = np.linspace(0.000001, 3,   num=303)
k       = np.logspace(np.log10(5e-5), np.log10(k_max), num=804)

# I'm not interested in varying the neutrino mass
#m_ncdm_variaz = np.zeros((15))
#percentuali = [-10, -5, -3.75, -2.5, -1.875, -1.250, -0.625, 0, 0.625, 1.250, 1.875, 2.5, 3.75, 5, 10]
#for i in range(15):
#    m_ncdm_variaz[i] = m_ncdm + (m_ncdm/100)*percentuali[i]
    
tabella = np.zeros((804, 303))
# P(k) is "vertical", for each z there is a column

#tabella[:, 0]  = k
#tabella[0, 1:] = z_array #xxx attention: the 0th column is already occupied

#XXXXXXXX PUO DARSI CHE IL LOG GLI DESSE FASTIDIO

#Create a params dictionary
#Need to specify the max wavenumber
 #UNITS: 1/Mpc questo da pag 38 IST

#
#k 804 punti logaritmicamente equispaziati nel range (5*10^-6, 50) in unità h/Mpc
#z 303 punti logaritmicamente equispaziati nel range (0, 2.5)
#
#poi costruisci tabella k, z, P(k,z) e interpoli 
#


#for m_ncdm in m_ncdm_variaz:
#    file = open("P(k,z)_m=%g.txt" %m_ncdm, "w+")                    

params = {
         'output':'mPk',
         'non linear':'halofit', #xxx punto delicato, takabird?
         'Omega_b':Omega_b,
         'Omega_cdm':Omega_cdm,
         'h':h,
         'sigma8':sigma8,
         'n_s':n_s,
         'P_k_max_1/Mpc':k_max, #units: 1/Mpc
         'z_max_pk': 2.038, 
         'N_ncdm': N_ncdm,
         'm_ncdm': m_ncdm,
         'N_ur': N_ur,
         'Omega_Lambda': Omega_Lambda, 
         'w0_fld': w0_fld,
         'wa_fld': wa_fld   
         }

#Initialize the cosmology andcompute everything
cosmo = Class()
cosmo.set(params)
cosmo.compute()

#Specify k and z
#così prende da k = 10^-5 a k = 10^log10(0.25) = kmax
#    i = 0
#    Pkz = np.zeros((804,2)) 
#    Pkz[:,0] = k

i=0 # 0th column is for the k values
for z in z_array:
    #Call these for the nonlinear and linear matter power spectra
    #  !usa lui xxx! 
    Pnonlin = np.array([cosmo.pk(ki, z) for ki in k]) #this is a 1d array   
    tabella[:,i] = Pnonlin #rows: different k; columns: different z
    # attention: first row is occupied as well!
    i += 1
    
np.save("%s/output/P(k,z).npy" %path, tabella)
    
     
    
        
        #Plin = np.array([cosmo.pk_lin(ki, z) for ki in k])xxx serve solo nonlin
    
        #Pkz[:,1] = Pnonlin 
     #   np.savetxt('P(k,z)%g.txt' %i, Pkz)
#        i = i+1
         

        
#plt.plot(k, Pnonlin, label='linear')


    
    
#plt.plot(k, Pnonlin, label='linear')

#xxx questo a me non serve
#NOTE: You will need to convert these to h/Mpc and (Mpc/h)^3
#to use in the toolkit. To do this you would do:
#k /= h
#Plin *= h**3
#Pnonlin *= h**3

#proviamo a plottare i dati da class:
#array = np.genfromtxt('explanatory_neutrinos10_z1_pk_cb.dat')
#k1 = array[:,0]
#P1 = array[:,1]
##rinormalizzo le unità di misura
#k1 = k1*h
#P1 = P1/(h**3)
#
#
#plt.plot(k1, P1, label = 'explanatory')
#plt.plot(k, Plin, label='linear')



