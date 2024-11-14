from flask import Flask, request, Response
from waitress import serve
import os
import argparse
import json

#Auth
from flask import Flask
from flask_httpauth import HTTPTokenAuth

import models.object_detector.yolo_detection as yd

from models.utils import *
import logging
import torch


#Logging
logging.basicConfig(filename='flask_logs/image_api.log',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

#Constants
port_num = 8066
prod = False
use_gpu = False
temp_dir = 'temp_images'
keep_files = False

tokens_path = 'cfg/tokens.json'
th_path = 'cfg/thresholds.json'

# Config paths
gr_weights_path = 'yolov3/weights/grounding.pt'
gr_names_path = 'yolov3/data/grounding.names'
gr_config_path = 'yolov3/cfg/yolov3-spp-1cls.cfg'

sd_weights_path = 'yolov3/weights/satelite_dish.pt'
sd_names_path = 'yolov3/data/satelite_dish.names'
sd_config_path = 'yolov3/cfg/yolov3-spp-1cls.cfg'

cj_weights_path = 'yolov3/weights/cable_jack.pt'
cj_names_path = 'yolov3/data/cable_jack.names'
cj_config_path = 'yolov3/cfg/yolov3-spp-1cls.cfg'

ad_weights_path = 'yolov3/weights/antenna.pt'
ad_names_path = 'yolov3/data/antenna.names'
ad_config_path = 'yolov3/cfg/yolov3-spp-1cls.cfg'

fe_weights_path = 'yolov3/weights/fire_extinguisher.pt'
fe_names_path = 'yolov3/data/fire_extinguisher.names'
fe_config_path = 'yolov3/cfg/yolov3-spp-1cls.cfg'

sn_weights_path = 'yolov3/weights/screw_nuts.pt'
sn_names_path = 'yolov3/data/screw_nuts.names'
sn_config_path = 'yolov3/cfg/yolov3-spp-2cls.cfg'


#Define app
app = Flask(__name__)

#Auth
auth = HTTPTokenAuth(scheme='Bearer')

#Load token dict from file
tokens = json.load(open(tokens_path))

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

# save_path - None means that the image is read from request using the Flask API, 
# othervise the save_path path is used to read the image from disk
# take_first - a flag used to extract the prediction of the first image
# temp_dir - it is a global variable pointing to the temp location for saving received images
def prediction_template(predictor, model_name, take_first=True, filter_list=[], save_path=None, **kwargs):
    """Template function for detection and response generation"""

#---------------------Save image part--------------------------

    # if save path is None get the image from the HTTP request using Flask API , save the image and set the multiple_models_mode flag
    # othervise use the save_path as it is for reading the image file from disk and set the multiple_models_mode flag
    if save_path == None:
        image = request.files.get('image')

        #Save image to a temporary location using save_image function from utils.py
        # image is an request flask object
        # model_name is a string used to generate descriptive file name
        # temp dir is the temporary directory at which the image will be saved
        save_path, time_str = save_image(image, model_name, temp_dir=temp_dir, app_logger=app.logger)
        multiple_models_mode = False
    else:
        multiple_models_mode = True
    

#---------------------Prediction part--------------------------
    #Process the image
    resp = predictor.predict(save_path, **kwargs)


#---------------------Prediction processing part--------------------------
    #Take first element
    if take_first:
        resp = json.loads(str(resp))
        resp = take_first_element(resp)

    #Filter classes
    if len(filter_list) > 0:
        resp = filter_response(resp, filter_list)
    
    #Add CI - add confidence interval to result based on the threshold found in the configuration file
    # it could be either low or high
    resp = add_confidence_interval(resp, save_path, model_name)

    #Remove saved temporary image
    if not keep_files and not multiple_models_mode:
        os.remove(save_path)

    #Return the response
    if not multiple_models_mode:
        #Not used in multiple_models
        return Response(resp)
    else:
        return resp

#Multiple Models
@app.route('/multiple_models', methods=['POST'])
@auth.login_required
def multiple_models():
    """Function for mutiple models in a single API call"""

    #Save image
    image = request.files.get('image')
    save_path, time_str = save_image(image, 'multiple_models', temp_dir=temp_dir, app_logger=app.logger)

    #Models
    #Get key words
    models = request.args.getlist('model')
    #No Models provided
    if len(models) == 0:
        #Log resp
        app.logger.info('Multiple Models:No Models provided')
        return Response(json.dumps([]))

    #Models dict
    models_dict = { 'grounding_detection':grounding_detection,
                    'satellite_dish_detection':satellite_dish_detection,
                    'cablejack_detection':cablejack_detection,
                    'antenna_detection':antenna_detection,
                    'fireextinguisher_detection':fireextinguisher_detection,
                    'screwnuts_detection':screwnuts_detection
                    }

    #Iterate over all models
    resp = dict()
    for model in models:
        #Add response
        resp[model] = json.loads(models_dict[model](save_path=save_path))

    #Delete file
    if not keep_files:
        os.remove(save_path)

    #Response
    return Response(json.dumps(resp))


#Models
# Grounding Detection
@app.route('/grounding_detection', methods=['POST'])
@auth.login_required
def grounding_detection(save_path=None):
    return prediction_template(ground_pred, 'grounding', take_first=True, save_path=save_path)

# Satellite Dish Detection
@app.route('/satellite_dish_detection', methods=['POST'])
@auth.login_required
def satellite_dish_detection(save_path=None):
    return prediction_template(satd_pred, 'satellite_dish', take_first=True, save_path=save_path)

# Cable Jack Detection
@app.route('/cablejack_detection', methods=['POST'])
@auth.login_required
def cablejack_detection(save_path=None):
    return prediction_template(cjack_pred, 'cable_jack', take_first=True, save_path=save_path)

# Antenna Detection
@app.route('/antenna_detection', methods=['POST'])
@auth.login_required
def antenna_detection(save_path=None):
    return prediction_template(antenna_pred, 'antenna_detection', take_first=True, save_path=save_path)

# Fire Extuinguisher Detection
@app.route('/fireextinguisher_detection', methods=['POST'])
@auth.login_required
def fireextinguisher_detection(save_path=None):
    return prediction_template(fireext_pred, 'fire_ext', take_first=True, save_path=save_path)

# Screw Nuts Detection
@app.route('/screwnuts_detection', methods=['POST'])
@auth.login_required
def screwnuts_detection(save_path=None):
    return prediction_template(screwnuts_pred, 'screw_nuts', take_first=True, save_path=save_path, filter_list=['double_nut'])

# Server Shutdown
def shutdown_server():
    if prod:
        return
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown_server_for_maintenance', methods=['POST'])
@auth.login_required
def shutdown_flask():
    shutdown_server()

    #Log shutdown
    app.logger.info('Shutdown Server')
    if not prod:
        return 'Server shutting down'
    else:
        return 'Shutdown method not available for Waitress'

if __name__ == '__main__':
    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port', type=int, default=8066, help='Port number for API hosting')
    parser.add_argument('-w','--prod', action='store_true', help='Serve using Waitress HTTP Server')
    parser.add_argument('-g','--use_gpu', action='store_true', help='Enable GPU usage')
    parser.add_argument('-t','--temp_dir', default='temp_images', help='Directory for storing temporary images')
    parser.add_argument('-k','--keep_files', action='store_true', help='Do not remove temporary image files')
    parser.add_argument('-n','--num_threads', type=int, default=4, help='Number of threads for Waitress')
    
    #Get args
    args = parser.parse_args()
    port_num = args.port
    prod = args.prod
    use_gpu = args.use_gpu
    temp_dir = args.temp_dir
    keep_files = args.keep_files
    num_threads = args.num_threads
    
    if use_gpu == False:
        # disable GPU devices
        torch.backends.cudnn.enabled = False
        torch.cuda.is_available = lambda : False
    
    # Initialize yolo models   
    ground_pred     = yd.YoloDetector(weights=gr_weights_path, names=gr_names_path, cfg=gr_config_path)
    satd_pred       = yd.YoloDetector(weights=sd_weights_path, names=sd_names_path, cfg=sd_config_path)
    cjack_pred      = yd.YoloDetector(weights=cj_weights_path, names=cj_names_path, cfg=cj_config_path)
    antenna_pred    = yd.YoloDetector(weights=ad_weights_path, names=ad_names_path, cfg=ad_config_path)
    fireext_pred    = yd.YoloDetector(weights=fe_weights_path, names=fe_names_path, cfg=fe_config_path)
    screwnuts_pred  = yd.YoloDetector(weights=sn_weights_path, names=sn_names_path, cfg=sn_config_path)

    if not prod:
        app.run(debug=False, port=port_num, threaded=False, host='0.0.0.0')
    else:
        serve(app, port=port_num,  host='0.0.0.0', threads=num_threads)