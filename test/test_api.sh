#!/bin/bash
#Set IP and AUth Header
IP="0.0.0.0"
PORT="8066"
HEADER="Bearer 9923psh2-76rw-1asd-6kuw-8993sa762kdl"

echo -e "Testing adress: "$IP" -> port: "$PORT"\n"

for i in {1..1}
do
	echo -e "Iteration: "$i

	# Grounding Detection
	echo "Grounding detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/grounding_detection" -F "image=@grounding.jpg"
	echo -e "\n"

	# Satelite Dish Detection
	echo "Satellite Dish detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/satellite_dish_detection" -F "image=@satellitedish.jpg"
	echo -e "\n"

	# Cable Jack Detection
	echo "Cable jack detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/cablejack_detection" -F "image=@cablejack.jpg"
	echo -e "\n"

	# Antenna Detection
	echo "Antenna detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/antenna_detection" -F "image=@antenna.jpg"
	echo -e "\n"

	# Fire Extinguisher Detection
	echo "Fire extinguisher detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/fireextinguisher_detection" -F "image=@fireextinguisher.jpg"
	echo -e "\n"

	# Screw Nuts Detection
	echo "Screw nuts detection: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/screwnuts_detection" -F "image=@screwnuts.jpg"
	echo -e "\n"

	# Multiple Models
	echo "Multiple models: "
	curl -H "Authorization":"$HEADER" -X POST "http://$IP:$PORT/multiple_models?model=grounding_detection&model=fireextinguisher_detection" -F "image=@fireextinguisher.jpg"
	echo -e "\n"
done