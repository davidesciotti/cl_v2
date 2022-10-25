import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys

matplotlib.use('Qt5Agg')

sys.path.append('/lib')
import my_module as mm

wig_dav = np.genfromtxt('/output/WF/WFs_v15_zNLA_gen22/wig_davide_multiBinBias_IST_nz10000.txt')
wil_dav = np.genfromtxt('/output/WF/WFs_v15_zNLA_gen22/wil_davide_IA_IST_nz10000_bia2.17.txt')
wig_dav = wig_dav[:, 1:]
wil_dav = wil_dav[:, 1:]

path_marco = '/config/common_data/everyones_WF_from_Gdrive/marco'
wil_marco = np.load(f'{path_marco}/wil_mar_bia2.17_IST_nz10000.npy').T
wig_marco = np.load(f'{path_marco}/wig_mar_IST_nz10000.npy').T
z_marco = np.linspace(0.001, 4, 10_000)


diff = mm.percent_diff_mean(wig_marco, wig_dav)

for i in range(10):
# i = 0
    plt.plot(z_marco, diff[:, i])
plt.axhline(-10)
plt.axhline(10)
plt.ylim(-10, 10)

