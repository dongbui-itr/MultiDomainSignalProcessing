import numpy as np
from matplotlib import pyplot as plt

start_time = 0
end_time = 1
sample_rate = 1000
time = np.arange(start_time, end_time, 1/sample_rate)
theta = 0
frequency = 100
amplitude = 1
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)

plt.plot(sinewave)
plt.show()
