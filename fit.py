import lmfit
import matplotlib.pyplot as plt
import numpy as np


def fitten_pulseheight(file, cutoff1, cutoff2):
    pulseheight_list = []
    counts = []
    with open(file) as input:
        next(input)

        # 
        for line in input:
            data_cut = line.split(",")
            pulseheight = float(data_cut[0])
            if pulseheight >= cutoff1 and pulseheight <= cutoff2 :
                pulseheight_list.append(float(data_cut[0]))
                counts.append(float(data_cut[2]))

    counts = np.array(counts)
    pulseheight = np.array(pulseheight_list)

    mod = lmfit.models.GaussianModel()
    params = mod.guess(counts, x=pulseheight)
    fit_result = mod.fit(counts, params, x=pulseheight)

    area = fit_result.params['amplitude'].value

    print(f"area= {area} ")

    uncertainty = 2*np.sqrt(2*np.log(2))*fit_result.params["fwhm"].value

    pulseheight_max = fit_result.params["center"].value
    print(f"the peak is at a pulsheight of {pulseheight_max} mV")

    plt.plot(pulseheight, counts)
    plt.plot(pulseheight, fit_result.best_fit)
    plt.title(file)
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


pulseheight_Na, std_Na, pulseheight_list_Na, counts_Na = fitten_pulseheight(
    "calibration_HPGE/spectrum_Na_22_2", 190,400
)
pulseheight_Cs1, std_Cs1, pulse_height_list_Cs1, counts_Cs1 = fitten_pulseheight(
    "calibration_HPGE/spectrum_Cs_137_1", 200,1000
)

pulseheight_Co2, std_C02, pulseheight_list_Co2, counts_Co2 = fitten_pulseheight(
    "calibration_HPGE/spectrum_Co_60_2", 500,1000
)

pulseheight_Co3, std_Co3, pulseheight_list_Co3, counts_Co3 = fitten_pulseheight(
    "spectrum_Co_60_3_semiconductor", 500,1000
)
pulseheight_array = np.array(
    [pulseheight_Na, pulseheight_Cs1, pulseheight_Co2, pulseheight_Co3]
)

Energy_list = [510.99, 662, 1332.5, 1332.5]
Energy_array = np.array(Energy_list)
pulseheight_fwhm = np.array([std_Na, std_Cs1, std_C02, std_Co3])


def linear_mod(x, a, b):
    return a * x + b


mod = lmfit.Model(linear_mod)
fit_result = mod.fit(
    Energy_array, a=5, b=0, x=pulseheight_array, weights=1 / pulseheight_fwhm
)
print(fit_result.fit_report())

a = fit_result.best_values["a"]
b = fit_result.best_values["b"]

print(f"a = {a}")
print(f"b = {b}")

x = np.linspace(0, 600, 100)
y = a * x + b

plt.errorbar(pulseheight_array, Energy_array, xerr=pulseheight_fwhm, fmt="o")
plt.plot(x, y)
plt.xlabel("pulseheight (mV)")
plt.ylabel("energy (keV)")
plt.xlim(0, 600)
plt.ylim(0, 1500)
plt.grid(True)
plt.show()


def fitten_Energy(list_pulseheight, list_counts, a, b):
    Energy_list = []
    counts = []
    for I in range(len(list_counts)):
        pulseheight = float(list_pulseheight[I])
        count = list_counts[I]
        Energy = a * pulseheight + b
        Energy_list.append(Energy)
        counts.append(count)

    counts = np.array(counts)
    Energy_list = np.array(Energy_list)

    mod = lmfit.models.GaussianModel()
    params = mod.guess(counts, x=Energy_list)
    fit_result = mod.fit(counts, params, x=Energy_list)

    uncertainty = fit_result.params["fwhm"].value

    energy_max = fit_result.params["center"].value
    print(f"the peak is at a energy of {energy_max} keV")

    plt.plot(Energy_list, counts)
    plt.plot(Energy_list, fit_result.best_fit)
    plt.xlabel("energy [keV]")
    plt.ylabel("counts")
    plt.grid(True)
    plt.show()

    return uncertainty, energy_max



fmwh_Na, energy_Na = fitten_Energy(pulseheight_list_Na, counts_Na, a, b)
resolution_Na = fmwh_Na / 510.99 * 100
print(f"resolution_Na_22_ = {resolution_Na} en fmwh_Na = {fmwh_Na}")

fmwh_Co_2, energy_Co_2 = fitten_Energy(pulseheight_list_Co2, counts_Co2, a, b)
resolution_Co2 = fmwh_Co_2 / 1173.2 * 100
print(f"resolution_Co_60_2 = {resolution_Co2} en fmwh_Co_2 = {fmwh_Co_2}")

fmwh_Cs_1, energy_Cs_1 = fitten_Energy(pulse_height_list_Cs1, counts_Cs1, a, b)
resolution_Cs1 = fmwh_Cs_1 / 662 * 100
print(f"resolution_Cs_137_1 = {resolution_Cs1} en fmwh_Cs_137_1 = {fmwh_Cs_1}")

fmwh_Co_3, energy_Co_3 = fitten_Energy(pulseheight_list_Co3, counts_Co3, a, b)
resolution_Cs1 = fmwh_Co_3 / 662 * 100
print(f"resolution_Cs_137_1 = {resolution_Cs1} en fmwh_Cs_137_1 = {fmwh_Cs_1}")


print(fit_result.fit_report())
a = fit_result.params['a'].value
a_std = fit_result.params['a'].stderr
b_std = fit_result.params['b'].stderr
cor_ab = fit_result.params['a'].correl['b']

# calculating std of the energy via propagation of error
def EnergyError(pulseheight, std_pulseheight):
    Energy_std = np.sqrt(pulseheight**2 * a_std**2 + b_std**2 + 2*pulseheight*cor_ab*a_std*b_std + a**2*std_pulseheight**2)
    return Energy_std

Energy_std = np.array([EnergyError(pulseheight_Na, std_Na),
                       EnergyError(pulseheight_Cs1, std_Cs1),
                       EnergyError(pulseheight_Co2, std_C02),
                       EnergyError(pulseheight_Co3, std_Co3)])

Energy = np.array([energy_Na, energy_Cs_1, energy_Co_2, energy_Co_3])

plt.errorbar(pulseheight_array, Energy, xerr = pulseheight_fwhm, yerr = Energy_std, fmt="o")
plt.plot(x, y)
plt.xlabel("pulseheight (mV)")
plt.ylabel("energy (keV)")
plt.xlim(0, 600)
plt.ylim(0, 1500)
plt.grid(True)
plt.show()


# opens theory values and makes a list of the isotopes and photopeaks
isotope_list = []
photo_peak_list = []
with open("tabel_3_gamma_ray_intensities.txt") as input:
        # skips the first four lines
        next(input)
        next(input)
        next(input)
        next(input)

        # adds all the photopeaks and isotopes in a list
        for line in input:
            data_cut = line.split("|")
            isotope = data_cut[0]
            photo_peak = float(data_cut[1])

            isotope_list.append(isotope)
            photo_peak_list.append(photo_peak)

# gives the range of the measured energies with the uncertainties
error_E_Na = EnergyError(pulseheight_Na, std_Na)
energy_Na_plus = energy_Na + error_E_Na
energy_Na_min = energy_Na - error_E_Na
print(f"the energy range is {energy_Na_min} - {energy_Na_plus}")

# finds the isotope of the source
possible_isotopes = []
for I in range(len(isotope_list)):
    if photo_peak_list[I] < energy_Na_plus and photo_peak_list[I] >= energy_Na_min:
        isotopes = isotope_list[I]
        possible_isotopes.append(isotopes)

print(f"the possible isotopes are {possible_isotopes}")



