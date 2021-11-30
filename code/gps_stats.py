# specify if horizontal surface velocity data are collected at GPS stations
from params import x,y,dx
import numpy as np

# Synthetic "GPS" station locations
gps_locs = np.zeros(np.shape(x))            # set to zero for no stations

#gps_locs[np.sqrt(x**2+y**2)<0.5*dx] = 1    # one station
gps_locs[:,] = 1                           # spatially continuous distribution

# # 9-station array example
# gps_locs[:,50,50] = 1
# gps_locs[:,60,50] = 1
# gps_locs[:,50,60] = 1
# gps_locs[:,60,60] = 1
# gps_locs[:,40,50] = 1
# gps_locs[:,50,40] = 1
# gps_locs[:,40,40] = 1
# gps_locs[:,40,60] = 1
# gps_locs[:,60,40] = 1
