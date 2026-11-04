import matplotlib.pyplot as plt
import lmfit
import numpy as np


pulseheight = []
counts = []

with open('spectrum_Na22_07') as input:
    next(input)

    for line in input:
        data_cut = line.split(",")
        pulseheight.append(float(data_cut[0]))
        counts.append(float(data_cut[1]))

counts = np.array(counts)
pulseheight =np.array(pulseheight)

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
