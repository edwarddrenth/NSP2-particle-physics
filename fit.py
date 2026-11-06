import lmfit
import matplotlib.pyplot as plt
import numpy as np


def fitten(file,cutoff):
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
    pulseheight =np.array(pulseheight_list)

    mod=lmfit.models.GaussianModel()
    params = mod.guess(counts, x= pulseheight)
    fit_result = mod.fit(counts, params, x= pulseheight)

    pulseheight_max = fit_result.params['center'].value
    print(f"the peak is at a pulsheight of {pulseheight_max} mV")

    plt.plot(pulseheight, counts)
    plt.plot(pulseheight,fit_result.best_fit)
    plt.xlabel('pulseheight [mV]')
    plt.ylabel('counts')
    plt.grid(True)
    plt.show()
    return pulseheight_max


# fitten('spectrum_Co_01')
# fitten('spectrum_Co_02')
# fitten('spectrum_Cs_01')
# fitten('spectrum_Cs_03')
# fitten('spectrum_Na22_02')
# fitten('spectrum_Na22_07')

pulseheight_list= [fitten('spectrum_Na22_07', 0), fitten('spectrum_Cs_01',200),fitten('spectrum_Cs_03',200), fitten('spectrum_Co_02',440) ]

Energy_list = [510.99, 662,662, 1332.5]
plt.scatter(Energy_list,pulseheight_list)
plt.xlabel('energy (kev)')
plt.ylabel('pulseheight (mV)')
plt.grid(True)
plt.show()