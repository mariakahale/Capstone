"""
Concurrent wind speeds (VEC) for Toronto freezing-rain events.

Source: "Estimating and Mapping Extreme Ice Accretion Hazard and Load Due
to Freezing Rain at Canadian Sites" (see reference.txt).

VEC[return_period][delta_t] -> concurrent wind speed
    return_period : 50 or 500 (years)
    delta_t       : 3, 7 or 21

USAGE:
    from concurrentwindspeed import VEC
    v_conc = VEC[50][3]   # -> 22
"""

VEC = {
    50:  {3: 22, 7: 21, 21: 23},   # VEC_50
    500: {3: 25, 7: 24, 21: 25},   # VEC_500
}


Va_50 = 27.6
concurrent_wind_speed_50 = VEC[50][7]  # concurrent wind speed for 50 yr return period, but delta_t = 7 days
#however, this is hourly mean.


#from eq.7 simplified (p. 135)
#Vec_t = Y_t / sqrt(1+ (2*A_mT/D))
#where D = diameter of bare wire.

#from Table 4
Y_50 = 31
A_m50_over_D = 0.52
#for D = 25 mm
Vec_50 = Y_50 / (1 + 2 * A_m50_over_D) ** 0.5

#but for D = 18.82 mm, we have to recalculate A_m50_over_D
#D = 18.82 mm is the diameter of the bare wire used in Ma et al. 2020 paper, which is different from the diameter of the bare wire used in Sheng's paper (D = 25 mm)
D = 18.82
A_m50 = 0.52 * 25  #from table 4, A_m50_over_D = A_m50 / D
Vec_50_newD = Y_50 / (1 + 2 * A_m50 / D) ** 0.5



##DURST STUFF
#Now, the Vec_50_newD is the concurrent wind speed for hourly mean at 10 m.
#However, we need it for 3s gust at 10 m, as used in Ma et al. 2020 paper.
#(Durst 1960, as reproduced in ASCE 7-16 Commentary Fig. C26.5-1)
conversion_factor = 1.52 #this is to convert hourly mean into 3s gust speed, as used in ma et all 2020
Vec_50_3s = Vec_50_newD * conversion_factor


#note: Krishnasamy and Kulendran (1998) recommend a concurrent wind of 0.55 of the 50-year extreme wind for Toronto. Sheng et al. (2023) report the equivalent ratio as R_EC/A-50 = V_EC-50 / V_A-50; for Toronto this gives 21/27.6 ≈ 0.76 (calculated from their Table 4).
