"""
Concurrent wind speed for Toronto, used to slice the Fig. 7 fragility
surfaces of Ma et al. (2020).

Source: Sheng, Tang & Hong (2023), "Estimating and Mapping Extreme Ice
Accretion Hazard and Load Due to Freezing Rain at Canadian Sites",
Int J Disaster Risk Sci 14:127-142, Table 4.

V_EC-T is their equivalent concurrent wind speed (Eq. 7). It follows the
Jones et al. (2002) method, the same method behind the ASCE Manual 74
concurrent-wind map that Ma et al. used for Grand Marais.
Values are hourly-mean speeds at 10 m (m/s).
"""

# V_EC[T][delta_t_days] for Toronto, Table 4 (CRREL ice model)
V_EC_HOURLY = {
    50:  {3: 22, 7: 21, 21: 23},
    500: {3: 25, 7: 24, 21: 25},
}
V_A50_HOURLY = 27.6   # 50-yr annual-max wind for Toronto, also listed in Table 4

DELTA_T = 7           # days; the value Sheng et al. use for their V_EC-50 map (Fig. 12)
RETURN_PERIOD = 50    # pairs with the 50-yr ice, as in ASCE 74 / Jones et al. (2002)

# Durst (1960), ASCE 7-16 Commentary Fig. C26.5-1:
# ratio of 3-s gust to hourly-mean speed, open terrain, 10 m
DURST_3S = 1.52


def concurrent_gust(T=RETURN_PERIOD, dt=DELTA_T):
    """Concurrent 3-s gust at 10 m (m/s): the units of the Fig. 7 wind axis."""
    return V_EC_HOURLY[T][dt] * DURST_3S


V_CONC = concurrent_gust()   # 21 * 1.52 = 31.9 m/s

if __name__ == "__main__":
    print(f"V_EC-50 (hourly, dt={DELTA_T} d): {V_EC_HOURLY[50][DELTA_T]} m/s")
    print(f"Concurrent 3-s gust for Fig. 7 slice: {V_CONC:.1f} m/s")
    print(f"Speed ratio V_EC-50 / V_A-50: {V_EC_HOURLY[50][DELTA_T] / V_A50_HOURLY:.2f}")
    print(f"Concurrent wind pressure ratio (V_EC-50 / V_A-50)^2: { (V_EC_HOURLY[50][DELTA_T] / V_A50_HOURLY)**2:.2f}")