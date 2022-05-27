k_array_2    = np.logspace(np.log10(5e-5), np.log10(30), num=470)
z_array_2 = np.linspace(0.000001, 2.7,   num=100)

vals = np.asarray([P(k_array[row], zi) for zi in z_array_2])




plt.plot(z_array, PS[row, :], label = "PS originale, z = %f" %z_array[column])
plt.plot(z_array_2, vals, label = "P(k,z) interpolato, z = %f" %z_array[column])
plt.ylabel("P(k, z)")
plt.title("Matter Power Spectrum")
plt.legend()
plt.grid("true", linestyle = "--")
#    plt.yscale("log")
plt.xlabel("$k$")