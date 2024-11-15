# Download project 
- clone project: `git clone git@github.com:closeoutusr/AI_microservice.git`
- download model weights using following links:
  - [antenna.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/ERNgQ6j2-zxLvfGlZj7hj9wBLdqg4lb8kAnmcMykwFwPaQ?e=Zm3kfa)
  - [cable_jack.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/Ea1R5FEoX71Ksa0Oc_fuigIBJoxDaoanmyvCHjin4sYcTQ?e=vAXD2b)
  - [fire_extinguisher.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/EYuRrrxK2a1CjsBbdLxi8ocBTbrdezmxoJSwqlmW2s32QQ?e=ihrH2B)
  - [grounding.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/Ee1ge4El8XxBqFZmI9bAWxkBL-H8LrTkOGbo2OQegH158A?e=eVqGoX)
  - [satelite_dish.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/EcDCmQVKNthDhXcLoo8DlWEBGS612ZtiXc7ZCrSkZd3Jng?e=1F9c6e)
  - [screw_nuts.pt](https://perunspark-my.sharepoint.com/:u:/g/personal/marius_baba_closeout_cloud/EU-BLdcD1tVPs6LYIfiRKzYBJMmIAR8h1RbRkJL_hbkHYg?e=IQHE94)
- copy downloaded weights to "./yolov3/weights" folder 

# Prepare the environment
- create new virtual environment
   - ensure you have virtualenv package installed. If not, you can install it using: `pip install virtualenv`
   - create a virtual environment with Python version 3.7: `virtualenv -p python3.7 myenv`
- activate virtual environment: `source myenv/bin/activate`
- install packages: `pip install -r ./requirements.txt`

# Run the service
Start the API by issuing the `python3 image_api.py` command. This ​​program accepts the following optional arguments:

 - -p PORT, --port PORT
   - port number for API hosting  
 - -w, --prod
   - serve using Waitress HTTP Server  
 - -g, --use_gpu
   - enable GPU usage  
 - -t TEMP_DIR, --temp_dir 	TEMP_DIR
   - directory for storing temporary images  
 - -k, --keep_files
   - do not remove temporary image files  
 - -n NUM_THREADS, --num_threads NUM_THREADS
   - number of threads for Waitress  

Default values for those arguments are:
- port 8066
- prod False
- use_gpu False
- temp_dir 'temp_images'
- keep_files False
- num_threads 4

# Test the service 
Run the shell script `./test/test_api.sh`.This test script contains requests for calling a single model as well as calling multiple models. It covers test for all AI models supported by the API, i.e. tests for:
 - Grounding Detection
 - Satellite Dish Detection
 - Cable Jack Detection
 - Antenna Detection
 - Fire Extuinguisher Detection
 - Screw Nuts Detection  
 
# Configuration files
**./cfg/thresholds.json** - contains thresholds for each model supported by the API.The format used to define thresholds is as follows:
- "model_name": {"th1":value, "th2":value}  
  - 'th1' is used in the prediction filtering process, predictions with confidence below th1 are ignored
  - 'th2' is reserved for future use

**./cfg/tokens.json** - contains the token used to access the API
