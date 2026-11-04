import matplotlib.pyplot as plt



pulseheight = []
counts = []

with open('spectrum_Na22_07') as input:
    next(input)

    for line in input:
        data_cut = line.split(",")
        pulseheight.append(float(data_cut[0]))
        counts.append(float(data_cut[1]))
        
plt.plot(pulseheight,counts)
plt.xlabel('pulseheight [mV]')
plt.ylabel('counts')
plt.grid(True)
plt.show()
