import numpy as np
from libc.math cimport exp
import cython

cdef double CALCIUM = 0.05
# cdef double CALCIUM = 0.15
cdef double EXOCYTOSIS_L = 0.60
# cdef double EXOCYTOSIS_L = 0.50
cdef double EXOCYTOSIS_PQ = 0.60
# cdef double EXOCYTOSIS_PQ = 0.50

# cdef double CALCIUM = 0
# cdef double EXOCYTOSIS_L = 0
# cdef double EXOCYTOSIS_PQ = 0


# ---------------------------- Boltzmann function -----------------------------
cdef double x_inf(double V, double V_x, double S_x):
    return 1/(1+exp(-(V-V_x)/S_x))
# -------------------------------- Bell function -------------------------------
cdef double tau_x(double V, double tau_V_x, double tau_0_x, double V_tau_x, double S_tau_x):
    return tau_V_x/(exp(-(V-V_tau_x)/S_tau_x)+exp((V-V_tau_x)/S_tau_x)) + tau_0_x

# ------------------------------------ CaL -------------------------------------
cdef double m_CaL_inf(double V):
    return x_inf(V, -30, 10)
cdef double h_CaL_inf(double V):
    return x_inf(V, -33, -5)
cdef double tau_m_CaL(double V):
    return tau_x(V, 1, 0.05, -23, 20)
cdef double tau_h_CaL(double V):
    return tau_x(V, 60, 51, 0, 20)
cdef double V_Ca = 65
cdef double g_CaL = 0.85

cdef double I_CaL(double V, double m_CaL, double h_CaL):
    return g_CaL*m_CaL**2*h_CaL*(V-V_Ca)

# ------------------------------------ CaPQ ------------------------------------
cdef double m_CaPQ_inf(double V):
    return x_inf(V, -1, 4)
cdef double h_CaPQ_inf(double V):
    return x_inf(V, -33, -5)
cdef double tau_m_CaPQ(double V):
    return tau_x(V, 1, 0.05, -23, 20)
cdef double tau_h_CaPQ(double V):
    return tau_x(V, 60, 51, 0, 20)
cdef double g_CaPQ = 0.35
cdef double I_CaPQ(double V, double m_CaPQ, double h_CaPQ):
    return g_CaPQ*m_CaPQ*h_CaPQ*(V-V_Ca)

# ------------------------------------ CaT -------------------------------------
cdef double m_CaT_inf(double V):
    return x_inf(V, -49, 4)
cdef double h_CaT_inf(double V):
    return x_inf(V, -52, -5)
cdef double tau_m_CaT(double V):
    return tau_x(V, 15, 0, -50, 12)
cdef double tau_h_CaT(double V):
    return tau_x(V, 20, 5, -50, 15)
cdef double g_CaT = 0.4
cdef double I_CaT(double V, double m_CaT, double h_CaT):
    return g_CaT*m_CaT**3*h_CaT*(V-V_Ca)

# ------------------------------------- Na -------------------------------------
cdef double m_Na_inf(double V):
    return x_inf(V, -30, 4)
cdef double h_Na_inf(double V):
    return x_inf(V, -52, -8)
cdef double tau_m_Na(double V):
    return tau_x(V, 6, 0.05, -50, 10)
cdef double tau_h_Na(double V):
    return tau_x(V, 120, 0.5, -50, 8)
cdef double V_Na = 70
cdef double g_Na = 11
cdef double I_Na(double V, double m_Na, double h_Na):
    return g_Na*m_Na**3*h_Na*(V-V_Na)

# ------------------------------------- K --------------------------------------
cdef double m_K_inf(double V):
    return x_inf(V, -25, 23)
cdef double tau_m_K(double V):
    return tau_x(V, 1.5, 15, -10, 25)
cdef double V_K = -75
cdef double g_K = 4.5
cdef double I_K(double V, double m_K):
    return g_K*m_K**4*(V-V_K)

# ------------------------------------- KA -------------------------------------
cdef double m_KA_inf(double V):
    return x_inf(V, -45, 10)
cdef double h_KA_inf(double V):
    return x_inf(V, -68, -10)
cdef double tau_m_KA(double V):
    return 0*V + 0.1
cdef double tau_h_KA(double V):
    return tau_x(V, 60, 5, 5, 20)
cdef double g_KA = 1
cdef double I_KA(double V, double m_KA, double h_KA):
    return g_KA*m_KA*h_KA*(V-V_K)

# ------------------------------------ KATP ------------------------------------
cdef double I_KATP(double V, double g_KATP):
    return g_KATP*(V-V_K)

# LEAK
cdef double V_L = -26
cdef double I_L(double V, double g_L):
    return g_L*(V-V_L)

# SOC
cdef double V_SOC = V_Ca
cdef double g_SOC = 0.03
cdef double I_SOC(double V):
    return g_SOC*(V-V_SOC)

# ---------------------------------- Geometry ----------------------------------
cdef double volume(double r_o, double r_i=0):
    cdef double pi = 3.1415926535
    return 4*pi*(r_o**3-r_i**3)/3 * 1e-15 # in liters

cdef double area(double r):
    cdef double pi = 3.1415926535
    return 4*pi*r**2

cdef double r_cell = 5.301
cdef double d_submem = 0.15
cdef double r_submem = r_cell - d_submem
cdef double r_ud = 0.05

cdef double Vol_c = volume(r_submem)
cdef double Vol_m = volume(r_cell, r_submem)
cdef double Vol_ud = volume(r_ud)/2

cdef double A_submem = area(r_submem)
cdef double A_cell = area(r_cell)
cdef double A_ud = area(r_ud)/2

cdef double D_Ca = 220e-3
cdef double B_ud = D_Ca*A_ud/(Vol_ud*1e15*r_ud)
cdef double B_m = D_Ca*A_submem/(Vol_c*1e15)

# ---------------------------------- Calcium -----------------------------------
cdef double f = 0.01
cdef double alpha = 5.18e-15
# cdef double Vol_ud = 2.618e-19
# cdef double Vol_m = 5.149e-14
# cdef double Vol_c = 5.725e-13
cdef double Vol_c_Vol_er = 31
cdef double N_PQ = 100
cdef double N_L = 400
# cdef double B_ud = 264
# cdef double B_m = 0.128
cdef double k_PMCA = 0.3
cdef double k_SERCA = 0.1
cdef double p_leak = 3e-4
cdef double n_PQ = 4
cdef double K_PQ = 2
cdef double n_L = 4
cdef double K_L = 50
cdef double n_m = 4
cdef double K_m = 2

cdef double i_CaPQ(double V):
    return g_CaPQ*(V-V_Ca)/N_PQ

cdef double CaPQ_0(double V, double CaM):
    return CaM - alpha*i_CaPQ(V)/(B_ud*Vol_ud)

cdef double i_CaL(double V):
    return g_CaL*(V-V_Ca)/N_L

cdef double CaL_0(double V, double CaM):
    return CaM - alpha*i_CaL(V)/(B_ud*Vol_ud)

# -------------------------------- Exocytosis ----------------------------------

cdef double f_H(double x, double K, double n):
    return x**n/(x**n + K**n)

cdef double GS_PQ(double V, double m_CaPQ, double h_CaPQ, double CaM, double f_cAMP):
    return (1-EXOCYTOSIS_PQ+EXOCYTOSIS_PQ*f_cAMP)*(m_CaPQ*h_CaPQ*f_H(CaPQ_0(V, CaM), K_PQ, n_PQ) + (1-m_CaPQ*h_CaPQ)*f_H(CaM, K_PQ, n_PQ))

cdef double GS_L(double V, double m_CaL, double h_CaL, double CaM, double f_cAMP):
    return (1-EXOCYTOSIS_L+EXOCYTOSIS_L*f_cAMP)*(m_CaL**2*h_CaL*f_H(CaL_0(V, CaM), K_L, n_L) + (1-m_CaL**2*h_CaL)*f_H(CaM, K_L, n_L))

cdef double GS_m(double CaM):
    return f_H(CaM, K_m, n_m)

cdef double GS(double V, double m_CaL, double h_CaL, double m_CaPQ, double h_CaPQ, double CaM, double f_cAMP):
    return GS_PQ(V, m_CaPQ, h_CaPQ, CaM, f_cAMP) + GS_L(V, m_CaL, h_CaL, CaM, f_cAMP) + GS_m(CaM)

# --------------------------- Differential equations ---------------------------
cdef double dV_dt(double V, double m_CaL, double h_CaL, double m_CaPQ, double h_CaPQ, double m_CaT, double h_CaT, double m_Na, double h_Na, double m_K, double m_KA, double h_KA, double g_KATP, double g_L, double f_cAMP):
    cdef double CaL = I_CaL(V, m_CaL, h_CaL)
    cdef double CaPQ = I_CaPQ(V, m_CaPQ, h_CaPQ)
    cdef double CaT = I_CaT(V, m_CaT, h_CaT)
    cdef double Na = I_Na(V, m_Na, h_Na)
    cdef double K = I_K(V, m_K)
    cdef double KA = I_KA(V, m_KA, h_KA)
    cdef double KATP = I_KATP(V, g_KATP)
    cdef double L = I_L(V, g_L)
    cdef double SOC = I_SOC(V)
    return -((1-CALCIUM+CALCIUM*f_cAMP)*(CaL+CaPQ+CaT)+Na+K+KA+KATP+L+SOC)/5

cdef double dm_CaL_dt(double V, double m_CaL):
    return (m_CaL_inf(V)-m_CaL)/tau_m_CaL(V)
cdef double dh_CaL_dt(double V, double h_CaL):
    return (h_CaL_inf(V)-h_CaL)/tau_h_CaL(V)
cdef double dm_CaPQ_dt(double V, double m_CaPQ):
    return (m_CaPQ_inf(V)-m_CaPQ)/tau_m_CaPQ(V)
cdef double dh_CaPQ_dt(double V, double h_CaPQ):
    return (h_CaPQ_inf(V)-h_CaPQ)/tau_h_CaPQ(V)
cdef double dm_CaT_dt(double V, double m_CaT):
    return (m_CaT_inf(V)-m_CaT)/tau_m_CaT(V)
cdef double dh_CaT_dt(double V, double h_CaT):
    return (h_CaT_inf(V)-h_CaT)/tau_h_CaT(V)
cdef double dm_Na_dt(double V, double m_Na):
    return (m_Na_inf(V)-m_Na)/tau_m_Na(V)
cdef double dh_Na_dt(double V, double h_Na):
    return (h_Na_inf(V)-h_Na)/tau_h_Na(V)
cdef double dm_K_dt(double V, double m_K):
    return (m_K_inf(V)-m_K)/tau_m_K(V)
cdef double dm_KA_dt(double V, double m_KA):
    return (m_KA_inf(V)-m_KA)/tau_m_KA(V)
cdef double dh_KA_dt(double V, double h_KA):
    return (h_KA_inf(V)-h_KA)/tau_h_KA(V)
cdef double dCaM_dt(double V, double m_CaL, double h_CaL, double m_CaPQ, double h_CaPQ, double m_CaT, double h_CaT, double CaM, double CaC):
    return -f*alpha*I_CaT(V, m_CaT, h_CaT)/Vol_m + f*N_PQ*Vol_ud/Vol_m*B_ud*m_CaPQ*h_CaPQ*(CaPQ_0(V, CaM)-CaM) + f*N_L*Vol_ud/Vol_m*B_ud*m_CaL**2*h_CaL*(CaL_0(V, CaM)-CaM) - f*Vol_c/Vol_m*k_PMCA*CaM - f*Vol_c/Vol_m*B_m*(CaM-CaC)
cdef double dCaC_dt(double CaM, double CaC, double CaER):
    return 10.0*f*(B_m*(CaM-CaC) + p_leak*(CaER-CaC) - k_SERCA*CaC)
cdef double dCaER_dt(double CaC, double CaER):
    return -f*Vol_c_Vol_er*(p_leak*(CaER-CaC) - k_SERCA*CaC)

cdef montefusco(double[::1] current_vals, double g_KATP, double g_L, double f_cAMP):
    V = current_vals[0]
    m_CaL = current_vals[1]
    h_CaL = current_vals[2]
    m_CaPQ = current_vals[3]
    h_CaPQ = current_vals[4]
    m_CaT = current_vals[5]
    h_CaT = current_vals[6]
    m_Na = current_vals[7]
    h_Na = current_vals[8]
    m_K = current_vals[9]
    m_KA = current_vals[10]
    h_KA = current_vals[11]
    CaM = current_vals[12]
    CaC = current_vals[13]
    CaER = current_vals[14]

    next_vals = np.zeros(15)
    next_currents = np.zeros(15)

    next_vals[0] = dV_dt(V, m_CaL, h_CaL, m_CaPQ, h_CaPQ, m_CaT, h_CaT, m_Na, h_Na, m_K, m_KA, h_KA, g_KATP, g_L, f_cAMP)
    next_vals[1] = dm_CaL_dt(V, m_CaL)
    next_vals[2] = dh_CaL_dt(V, h_CaL)
    next_vals[3] = dm_CaPQ_dt(V, m_CaPQ)
    next_vals[4] = dh_CaPQ_dt(V, h_CaPQ)
    next_vals[5] = dm_CaT_dt(V, m_CaT)
    next_vals[6] = dh_CaT_dt(V, h_CaT)
    next_vals[7] = dm_Na_dt(V, m_Na)
    next_vals[8] = dh_Na_dt(V, h_Na)
    next_vals[9] = dm_K_dt(V, m_K)
    next_vals[10] = dm_KA_dt(V, m_KA)
    next_vals[11] = dh_KA_dt(V, h_KA)
    next_vals[12] = dCaM_dt(V, m_CaL, h_CaL, m_CaPQ, h_CaPQ, m_CaT, h_CaT, CaM, CaC)
    next_vals[13] = dCaC_dt(CaM, CaC, CaER)
    next_vals[14] = dCaER_dt(CaC, CaER)

    next_currents[0] = I_CaL(V, m_CaL, h_CaL)
    next_currents[1] = I_CaPQ(V, m_CaPQ, h_CaPQ)
    next_currents[2] = I_CaT(V, m_CaT, h_CaT)
    next_currents[3] = I_Na(V, m_Na, h_Na)
    next_currents[4] = I_K(V, m_K)
    next_currents[5] = I_KA(V, m_KA, h_KA)
    next_currents[6] = I_KATP(V, g_KATP)
    next_currents[7] = I_L(V, g_L)
    next_currents[8] = I_SOC(V)
    next_currents[9] = CaL_0(V, CaM)
    next_currents[10] = CaPQ_0(V, CaM)
    next_currents[11] = GS_L(V, m_CaL, h_CaL, CaM, f_cAMP)
    next_currents[12] = GS_PQ(V, m_CaPQ, h_CaPQ, CaM, f_cAMP)
    next_currents[13] = GS_m(CaM)
    next_currents[14] = GS(V, m_CaL, h_CaL, m_CaPQ, h_CaPQ, CaM, f_cAMP)

    return next_vals, next_currents

def montefusco_euler(double[::1] initial_vals, double g_KATP, double g_L, double f_cAMP, double t, double dt):
    cdef int i
    cdef int N = int(t/dt)
    v = np.zeros((N, 15))
    c = np.zeros((N, 15))
    v[0,:] = initial_vals
    for i in range(1,N):
        next_vals, next_currents = montefusco(v[i-1], g_KATP, g_L, f_cAMP)
        v[i] = v[i-1] + dt*next_vals
        c[i-1] = next_currents
    c[-1] = c[-2]
    return v, c
