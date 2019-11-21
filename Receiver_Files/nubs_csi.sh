#!/bin/sh

python3.6 nubs_amp.py
echo "Predicting user:"
python3.6 "nubs/detect_csi.py"
echo "Prediction done."
