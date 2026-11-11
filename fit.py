import lmfit
import matplotlib.pyplot as plt
import numpy as np


def fitten_pulseheight(file, cutoff):
    pulseheight_list = []
    counts = []
    with open(file) as input:
        next(input)

        for line in input:
            data_cut = line.split(",")
            pulseheight = float(data_cut[0])
            if pulseheight >= cutoff:
                pulseheight_list.append(float(data_cut[0]))
                counts.append(float(data_cut[1]))

    counts = np.array(counts)
    pulseheight = np.array(pulseheight_list)

    mod = lmfit.models.GaussianModel()
    params = mod.guess(counts, x=pulseheight)
    fit_result = mod.fit(counts, params, x=pulseheight)

    uncertainty = fit_result.params["fwhm"].value

    pulseheight_max = fit_result.params["center"].value
    print(f"the peak is at a pulsheight of {pulseheight_max} mV")

    plt.plot(pulseheight, counts)
    plt.plot(pulseheight, fit_result.best_fit)
    plt.xlabel("pulseheight [mV]")
    plt.ylabel("counts")
    plt.grid(True)
    plt.show()
    return pulseheight_max, uncertainty, pulseheight_list, counts


# fitten('spectrum_Co_01')
# fitten('spectrum_Co_02')
# fitten('spectrum_Cs_01')
# fitten('spectrum_Cs_03')
# fitten('spectrum_Na22_02')
# fitten('spectrum_Na22_07')


pulseheight_Na, fwhm_Na, pulseheight_list_Na, counts_Na = fitten_pulseheight(
    "spectrum_Na22_07", 0
)
pulseheight_Cs1, fwhm_Cs1, pulse_height_list_Cs1, counts_Cs1 = fitten_pulseheight(
    "spectrum_Cs_01", 200
)
pulseheight_Cs3, fwhm_Cs3, pulseheight_list_Cs3, counts_Cs3 = fitten_pulseheight(
    "spectrum_Cs_03", 200
)
pulseheight_Co2, fwhm_C02, pulseheight_list_Co2, counts_Co2 = fitten_pulseheight(
    "spectrum_Co_02", 440
)
pulseheight_array = np.array(
    [pulseheight_Na, pulseheight_Cs1, pulseheight_Cs3, pulseheight_Co2]
)

Energy_list = [510.99, 662, 662, 1332.5]
Energy_array = np.array(Energy_list)
pulseheight_fwhm = np.array([fwhm_Na, fwhm_Cs1, fwhm_Cs3, fwhm_C02])


def linear_mod(x, a, b):
    return a * x + b


mod = lmfit.Model(linear_mod)
fit_result = mod.fit(
    Energy_array, a=5, b=0, x=pulseheight_array, weights=1 / pulseheight_fwhm
)

a = fit_result.best_values["a"]
b = fit_result.best_values["b"]

print(f"a = {a}")
print(f"b = {b}")

x = np.linspace(0, 500, 100)
y = a * x + b

plt.errorbar(pulseheight_array, Energy_array, xerr=pulseheight_fwhm, fmt="o")
plt.plot(x, y)
plt.xlabel("pulseheight (mV)")
plt.ylabel("energy (keV)")
plt.xlim(0, 500)
plt.ylim(0, 1500)
plt.grid(True)
plt.show()

