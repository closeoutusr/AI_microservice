import numpy as np
import datetime
import os
import json
from PIL import Image
import cv2


def save_image(image, prefix, temp_dir, app_logger, debug=False):
    """Save image"""
    #Time str
    rand_ext = np.random.randint(10e6)
    time_str = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S_%f") + '_' + str(rand_ext)
    
    #Extension
    suffix = image.filename.split('.')[-1].lower()
    
    #For debug
    if debug:
        app_logger.info('Received File Name: {}'.format(image.filename))
    
    #Save image
    save_path = os.path.join(temp_dir, '{}_{}.{}'.format(prefix, time_str, suffix))
    image.save(save_path)
            
    return save_path, time_str

def take_first_element(resp):
    """Reformat Yolo results"""
    keys = list(resp.keys())
    if len(keys) > 0:
        return json.dumps(resp[keys[0]])
    else:
        return json.dumps([])

def load_parameter_from_config(config_path, param):
    """Return value of a parameter from config file"""
    with open(config_path, 'r') as f:
        params = json.load(f)
    return params[param]

def add_ci(c, th1, th2):
    """Add Confidence Interval"""
    if c < th1:
        return 'low'
#     elif c < th2:
#         return 'mid'
    else:
        return 'high'

def add_confidence_interval(resp, img_path, model, cfg_path='cfg/thresholds.json', rotate_boxes=True):
    """Add low/high, optionally rotate boxes"""
    objects = json.loads(resp)
    if len(objects) == 0:
        return resp
    else:
        #Load params
        th_dict = load_parameter_from_config(cfg_path, model)
        #Add CI for each object
        new_objects = []
        #For BB rotation
        if rotate_boxes:
            exif_rotation = get_image_exif_rotation(img_path)
            img_shape = cv2.imread(img_path).shape
        for obj in objects:
            c = float(obj['conf'])
            obj['conf_interval'] = add_ci(c, th_dict['th1'], th_dict['th2'])
            #Rotate BBs
            if 'coordinates' in obj:
                if rotate_boxes and len(obj['coordinates']) != 0:
                    obj['coordinates'] = rotate_box(obj['coordinates'], exif_rotation, img_shape)
            #Filter low conf
            if obj['conf_interval'] != 'low':
                new_objects.append(obj)
        #Return objects with CI
        return json.dumps(new_objects)


def filter_response(resp, classes_to_keep):
    """Filter low conf objects"""
    objects = json.loads(resp)

    #Filtered objects
    filtered_objects = [obj for obj in objects if obj['class'] in classes_to_keep]

    #Return filtered
    return json.dumps(filtered_objects)

def get_image_exif_rotation(image):
    """Get image exif rotation"""
    img = Image.open(image)
    exif_data = img._getexif()
    if not exif_data:
        return 0
    if not 274 in exif_data.keys():
        return 0
    else:
        return exif_data[274]

def minmax_to_all_corners(x_min, y_min, x_max, y_max, clockwise=False):
        """Converts (x1,y1,x2,y2) to four points."""
        if clockwise:
            new_poly = [[x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]]
        else:
            new_poly = [[x_min, y_min], [x_min, y_max],
                        [x_max, y_max], [x_max, y_min]]
        return new_poly

def rotate_box(box, exif_rotation, img_shape):
    """Bounding Box(BB) rotation for translating four image BBs back to the original image."""
    #Height/width
    img_width = img_shape[1]
    img_height = img_shape[0]

    #Get min/max x/y
    x_min = min(float(box[0][0]), float(box[1][0]), float(box[2][0]), float(box[3][0]))
    x_max = max(float(box[0][0]), float(box[1][0]), float(box[2][0]), float(box[3][0]))

    y_min = min(float(box[0][1]), float(box[1][1]), float(box[2][1]), float(box[3][1]))
    y_max = max(float(box[0][1]), float(box[1][1]), float(box[2][1]), float(box[3][1]))

    if exif_rotation == 0 or exif_rotation == 1:
        new_x_min = x_min
        new_x_max = x_max
        new_y_min = y_min
        new_y_max = y_max
    elif exif_rotation == 6:
        new_x_min = y_min
        new_x_max = y_max
        new_y_min = img_width - x_max
        new_y_max = img_width - x_min
    elif exif_rotation == 3:
        new_x_min = img_width - x_max
        new_x_max = img_width - x_min
        new_y_min = img_height - y_max
        new_y_max = img_height - y_min
    elif exif_rotation == 8:
        new_x_min = img_height - y_max
        new_x_max = img_height - y_min
        new_y_min = x_min
        new_y_max = x_max

    return minmax_to_all_corners(new_x_min, new_y_min, new_x_max, new_y_max)        