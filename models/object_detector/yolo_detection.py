import os
import json
import cv2

from yolov3.models import *
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
import torch


class YoloDetector:

    def __init__(self,
                 weights,
                 names,
                 cfg,
                 mode='str',
                ):
        
        self._mode = mode
        self._weights = weights
        self._cfg = cfg
        self._names = names

        
    def predict(self, path,
                save_img=False, save_txt=False, output_path='test_data/output',
                yolo_config='models/object_detector/config/yolo_detection_config.json',
                video = False,
                video_sample_rate = 1
               ):
        
        # parse config file that contains parameters for yolo detection process
        with open(yolo_config, 'r') as f:
            data = json.load(f)
            img_size = tuple(int(x) for x in data['img_size'][1:-1].split(','))
            half = data['half'] == 'True'
            conf_thres = float(data['conf_thres'])
            iou_thres = float(data['iou_thres'])
            classes_filter = None if data['classes_filter'] == 'None' else data['classes_filter']
            agnostic_nms = data['agnostic_nms'] == 'True'
            augment = data['augment'] == 'True'
            device = data['device']
            input_type = data['input_type']
            fourcc = data['fourcc']
#             print(img_size, half, conf_thres, iou_thres, classes_filter, augment, agnostic_nms, device)
        
        with torch.no_grad():
            img_size = 608  # if ONNX_EXPORT else img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

            # Initialize
            if ONNX_EXPORT or ("CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "-1"):
                device = 'cpu'
            
            device = torch_utils.select_device()
            if save_img or save_txt:
                if os.path.exists(output_path):
                    shutil.rmtree(output_path)  # delete output folder
                os.makedirs(output_path)  # make new output folder

            # Initialize model
            model = Darknet(self._cfg, img_size)

            # Load weights
            attempt_download(self._weights)
            if self._weights.endswith('.pt'):  # pytorch format
                model.load_state_dict(torch.load(self._weights, map_location=device)['model'])
            else:  # darknet format
                load_darknet_weights(model, self._weights)

            # Eval mode
            model.to(device).eval()

            # Fuse Conv2d + BatchNorm2d layers
            # model.fuse()

            # Export mode
            if ONNX_EXPORT:
                model.fuse()
                img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
                f = self._weights.replace(self._weights.split('.')[-1], 'onnx')  # *.onnx filename
                torch.onnx.export(model, img, f, verbose=False, opset_version=11)

                # Validate exported model
                import onnx
                model = onnx.load(f)  # Load the ONNX model
                onnx.checker.check_model(model)  # Check that the IR is well formed
                print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
                return

            # Half precision
            half = half and device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                model.half()

            # Set Dataloader
            vid_path, vid_writer = None, None
            view_img = False
            if input_type == 'webcam':
                path = 0
                view_img = True
                torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(path, img_size=img_size)
            else:
                dataset = LoadImages(path, img_size=img_size)

            # Get names and colors
            names = load_classes(self._names)
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            t0 = time.time()

            results = {}

            #Calculate sample freq
            if video:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                sample_freq = np.ceil(fps / video_sample_rate)
                cap.release()
            else:
                sample_freq = 1
            #Frame
            frame = -1
                
            for path, img, im0s, vid_cap in dataset:
                
                #Frame
                frame += 1
                
                #Process every nth frame
                if video and frame % sample_freq != 0:
                    continue
                
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = torch_utils.time_synchronized()
                pred = model(img, augment=augment)[0]
                t2 = torch_utils.time_synchronized()

                # to float
                if half:
                    pred = pred.float()

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, multi_label=False,
                                           classes=classes_filter, agnostic=agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if input_type == 'webcam':  # batch_size >= 1
                        p, s, im0 = path[i], '%g: ' % i, im0s[i]
                    else:
                        p, s, im0 = path, '', im0s

                    save_path = str(Path(output_path) / Path(p).name)
                    s += '%gx%g ' % img.shape[2:]  # print string
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in det:
                            if save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                            if save_img or view_img:  # Add bbox to image
                                label = '%s %.2f' % (names[int(cls)], conf)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                            x0, y0 = int(xyxy[0]), int(xyxy[1])
                            x1, y1 = int(xyxy[0]), int(xyxy[3])
                            x2, y2 = int(xyxy[2]), int(xyxy[3])
                            x3, y3 = int(xyxy[2]), int(xyxy[1])
                            
                            to_add = {
                                    'class': names[int(cls)],
                                    'conf': str(float(conf)),
                                    'coordinates': [[str(x0), str(y0)],
                                                    [str(x1), str(y1)],
                                                    [str(x2), str(y2)],
                                                    [str(x3), str(y3)]]                            
                                }
                            if video:
                                to_add['frame'] = str(frame)
                            
                            
                            if path in results:
                                results[path].append(to_add)
                            else:
                                results[path] = [to_add]

                    # Print time (inference + NMS)
#                    print('%sDone. (%.3fs)' % (s, t2 - t1))

                    # Stream results
                    if view_img:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'images':
                            cv2.imwrite(save_path, im0)
                        else:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer

                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                            vid_writer.write(im0)
                
            if len(results.keys()) == 0:
                results[path] = []
                    
            if save_txt or save_img:
                print('Results saved to %s' % os.getcwd() + os.sep + output_path)
#                 if platform == 'darwin':  # MacOS
#                     os.system('open ' + output_path + ' ' + save_path)

            print('Done. (%.3fs)' % (time.time() - t0))
        #Empty cache
        torch.cuda.empty_cache()
        return json.dumps(results)