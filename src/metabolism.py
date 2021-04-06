import numpy as np
from scipy.optimize import bisect, root

# --------------------------------- PARAMETERS ---------------------------------
J_max = 7.2
K_m = 5
p_L = 0.9
p_TCA = 1
J_O2_0 = 16
A_tot = 4000
ag_K_ATP = 54
g_s_2 = 0.22
g_K_ATP_ms = 0.27
n_s = 10
RGS_0 = 0.03

# Grubelnik et al 2020
k_ATPase = 135
K_m_ATPase = 2000

# cAMP
a_ATPase = 0
b_ATPase = 0.025


alpha_CO2 = 0.0308
pKa = 6.1

CO2_bas = 40*0.0308*1e-3
H_bas = 10**(-7.4)
HCO3_bas = CO2_bas*10**(-np.log10(H_bas) - pKa)

Ka = 10**(-pKa)
k_HCO3_PM = 1
k_H_PM = 1e6
k_CO2_PM = 1e3

Km_sAC = 11
Km_PDE3B = 0.4
Km_PDE4 = 4.4

r_sAC = 1.2
r_PDE3B = 0.5
r_PDE4 = 0.5

# ---------------------------- METABOLIC FUNCTIONS -----------------------------

# GLYCOLYSIS
J_G6P = lambda G: J_max * G**2 / (K_m**2 + G**2)
J_ATP_Gly = lambda G: 2*J_G6P(G)
J_NADH_Gly = lambda G: 2*J_G6P(G)
J_pyr = lambda G: 2*J_G6P(G)
J_ATP_NADH_Gly = lambda G: 1.5*(J_NADH_Gly(G)-p_L*J_pyr(G))

# TCA CYCLE
J_NADH_pyr = lambda G: 5*p_TCA*(1-p_L)*J_pyr(G)
J_ATP_NADH_pyr = lambda G: 2.5*J_NADH_pyr(G)

# OXYGEN INPUT
J_O2_G = lambda G: 0.5*(J_NADH_pyr(G)+J_NADH_Gly(G)-p_L*J_pyr(G))
J_O2 = lambda G: -0.2*J_G6P(G) + J_O2_0

# BETA-OXIDATION
J_NADH_FFA = lambda G: 2*(J_O2(G)-J_O2_G(G))
J_ATP_NADH_FFA = lambda G: 2.3*J_NADH_FFA(G)

# ATP INPUT
J_ATP = lambda G: J_ATP_Gly(G) + J_ATP_NADH_Gly(G) + J_ATP_NADH_pyr(G) + J_ATP_NADH_FFA(G)

# ATP OUTPUT
J_ATPase = lambda ATP: b_ATPase*ATP + a_ATPase

# AXP CONCENTRATIONS
ATP = lambda G: (J_ATP(G)-a_ATPase)/b_ATPase
ADP = lambda G: A_tot - ATP(G)
RAT = lambda G: ATP(G)/ADP(G)

# K(ATP) CHANNELS
# g_K_RAT = lambda RAT: 0.56*np.exp(-0.25*RAT)

def f(x):
    A, B = x
    return [A*np.exp(-B*RAT(1))+0.19-0.27, A*np.exp(-B*RAT(6))+0.19-0.2]
from scipy.optimize import root
res = root(f, [0, 0]).x

g_K_RAT = lambda RAT: res[0]*np.exp(-res[1]*RAT)+0.19

def glucose(gkatp):
    f = lambda g: g_K_RAT(RAT(g))-gkatp
    return bisect(f, 0, 100)

# CO2 OUTPUT
J_CO2 = lambda G: 3*p_TCA*(1-p_L)*J_pyr(G) + 0.7/2*J_NADH_FFA(G)
# Lactate output
J_lac = lambda G: p_L*J_pyr(G)

# ---------------------- BICARBONATE BUFFER FUNCTIONS --------------------------

J_H_PM = lambda k_H, H_in: k_H*(H_in-H_bas)
J_CO2_PM = lambda k_CO2, CO2_in: k_CO2*(CO2_in-CO2_bas)
J_HCO3_PM = lambda k_HCO3, HCO3_in: k_HCO3*(HCO3_bas-HCO3_in)

def equations(G, const, p):
    HCO3_in, H_in, CO2_in, J_IC = p
    k_H, k_CO2, k_HCO3 = const
    return [
        J_lac(G) - J_H_PM(k_H, H_in) - J_IC,
        J_CO2(G) - J_CO2_PM(k_CO2, CO2_in) + J_IC,
        J_HCO3_PM(k_HCO3, HCO3_in) - J_IC,
        10**(-pKa) - H_in*HCO3_in/CO2_in
    ]

def intracellular_concentrations(G, const):
    return root(lambda a: equations(G, const, a), [HCO3_bas, H_bas, CO2_bas, 0], method="lm").x

pars = (k_HCO3_PM, k_H_PM, k_CO2_PM)
HCO3_in = lambda G: intracellular_concentrations(G, pars)[0]*1000  # mM
H_in = lambda G: intracellular_concentrations(G, pars)[1]  # M
CO2_in = lambda G: intracellular_concentrations(G, pars)[2]*1000  # mM

# ------------------------------- cAMP FUNCTIONS -------------------------------

J_sAC = lambda HCO3, r_sAC: r_sAC*HCO3/(HCO3+Km_sAC)
J_PDE = lambda cAMP, r_PDE3B, r_PDE4: r_PDE3B*cAMP/(cAMP+Km_PDE3B) + r_PDE4*cAMP/(cAMP+Km_PDE4)

def cAMP(hco3):
    f = lambda camp: J_sAC(hco3, r_sAC)-J_PDE(camp, r_PDE3B, r_PDE4)
    return bisect(f, 0, 100)

fcAMP = lambda G: (cAMP(HCO3_in(G)) - cAMP(HCO3_in(20)))/(cAMP(HCO3_in(0))-cAMP(HCO3_in(20)))


# ---------------------------- GLUCAGON SECRETION ------------------------------

f_RGS = lambda g_K_ATP: g_K_ATP**n_s/(g_s_2**n_s + g_K_ATP**n_s)
RGS = lambda g_K_ATP: (1-RGS_0)*f_RGS(g_K_ATP)/f_RGS(g_K_ATP_ms)+RGS_0
