import numpy as np
import matplotlib.pyplot as plt

import src.metabolism as alpha
from src.secretion_cy import montefusco_euler

try:
    [glucose, mean_V, mean_GS] = np.loadtxt("data.txt")
except OSError:
    # Defining the "leak" current
    g_L = lambda g_K_ATP: 0.2 + (0.25**9/(0.25**9+g_K_ATP**9))*0.04

    # Defining initial values
    t = 200000
    dt = 0.1
    x_0 = np.array([-5.14063726e+01,  1.05155470e-01,  7.76130678e-01,
                    3.36181980e-06, 7.76130678e-01,  3.68483231e-01,
                    3.97062254e-01, 4.67787961e-03, 2.09233487e-01,
                    2.81686443e-01,  3.45055157e-01, 1.54643301e-01,
                    2.37897318e-01,  2.94083254e-01,  9.85403353e+01]
                   )

    # Defining parameters
    mean_V = []
    mean_GS = []
    glucose = np.linspace(0, 15, 30)
    for g in glucose:
        print(g)
        # Calculating gKATP from metabolic model (metabolic part)
        gkatp = alpha.g_K_RAT(alpha.RAT(g))

        # Calculating fcAMP from metabolic model (signaling part)
        fcAMP = alpha.fcAMP(g, 1.2, 0.8, 0.2)

        # Calculating secretion parameters from the voltage model
        voltage, currents = montefusco_euler(x_0, gkatp, g_L(gkatp), fcAMP,
                                             t, dt
                                             )
        mean_V.append(np.mean(voltage[int(4*t/5):, 0]))
        mean_GS.append(np.mean(currents[int(4*t/5):, -1]))
    np.savetxt("data.txt", np.array([glucose, mean_V, mean_GS]))

# Plotting results
fig, ax = plt.subplots()
ax.plot(glucose, mean_GS/mean_GS[0])
ax.set_xlabel("Glucose concentration (mM)")
ax.set_ylabel("RGS (%)")
plt.show()
