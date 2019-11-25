#!/usr/bin/python3.6

import matlab.engine
import time

#from nubs import detect_csi

print("Converting .dat to .mat")
eng = matlab.engine.start_matlab()
eng.nubs_amplitude_gen(nargout=0)
eng.quit()
print("Conversion done.")

# do other shit here
