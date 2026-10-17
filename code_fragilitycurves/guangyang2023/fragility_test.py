import guangyang_fragility_lookup as gfl
x = gfl.lookup_wind_fragility_class4_pole(50)
print(f"Probability of failure for Class 4 pole at wind speed 21.1 m/s: {x:.4f}")