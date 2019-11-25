#!/bin/sh

arg=$1

echo "    -Starting server.."
gnome-terminal -e "python3.6 nubs_app.py"
echo "    -Server started"

cp $arg nubs/nubs_data_in/
mv nubs/nubs_data_in/$arg nubs/nubs_data_in/log_output.dat

#timestamp=$(date +%s)

# Convert the CSI data to amplitude-time domain
echo "    -Preprocessing data..."
python3.6 nubs_amp.py
echo "    -Predicting user:"


python3.6 -W ignore "nubs/detect_csi.py" $arg

# Process and apply the CSI data to our model
#person=`python3.6 -W ignore "nubs/detect_csi.py" | tail -n 1`
#echo "    -Prediction done."

#person="Vintony"
# Send a post request using the predicted person to WebAPP.
#curl -X POST -H 'Content-Type: application/json' -d "{\"person\":\"$person\",\"time\":\"$timestamp\"}" http://localhost:5000/csi/new
#echo "    -Data sent to WebAPP."
