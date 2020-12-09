from argparse import ArgumentParser
from collections import defaultdict
import math
import sys, os

import cv2
sys.path.append(os.path.join(os.getcwd(),'darknet/python/'))
import darknet as dn
import numpy as np


DETECT_THRESH = .7
MIN_IOU       = .2 # TODO figure out what this should be

def _get_data_raw(data_path):
    leafs = sorted(os.listdir(data_path))
    full = map(lambda d: os.path.join(data_path, d), leafs)
    return map(lambda d: d.encode('ascii'), full)


def _prep_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    existing_files = os.listdir(output_path)
    if len(existing_files) > 0:
        print(f'Deleting files in {output_path}')
        for f in existing_files:
            p = os.path.join(output_path, f)
            os.remove(p)


def _detect_people(net, meta, frame):
    def filter_people_detections(detection):
        return detection[0] == b'person' and detection[1] > DETECT_THRESH

    r = dn.detect(net, meta, frame)
    return list(filter(filter_people_detections, r))


def _test_detector(net, meta, data, output_path):
    for d in data:
        pstr = d.decode('ascii')
        out = os.path.join(output_path, os.path.basename(pstr))
        base = cv2.imread(os.path.join(os.getcwd(), pstr))

        people_detections = _detect_people(net, meta, d)

        for detection in people_detections:
            cx, cy, w, h = detection[2]
            x, y = cx - (w/2), cy - (h/2)
            x, y, w, h = int(x), int(y), int(w), int(h)
            base = cv2.rectangle(base, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imwrite(out, base)


def _iou(bb1, bb2):
    bb1_cx, bb1_cy, bb1_w, bb1_h = bb1
    bb2_cx, bb2_cy, bb2_w, bb2_h = bb2

    bb1_l, bb1_t = bb1_cx-(bb1_w/2), bb1_cy-(bb1_h/2)
    bb1_r, bb1_b = bb1_l+bb1_w, bb1_t+bb1_h

    bb2_l, bb2_t = bb2_cx-(bb2_w/2), bb2_cy-(bb2_h/2)
    bb2_r, bb2_b = bb2_l+bb2_w, bb2_t+bb2_h

    # intersection
    il, it = max(bb1_l, bb2_l), max(bb1_t, bb2_t)
    ir, ib = min(bb1_r, bb2_r), min(bb1_b, bb2_b)
    ia = max(0, ir-il)*max(0,ib-it)
    # union
    ua = ((bb1_r-bb1_l)*(bb1_b-bb1_t)) + ((bb2_r-bb2_l)*(bb2_b-bb2_t)) - ia

    return ia/ua


def _state_to_bb(state):
    cx, cy, s, r, _, __, ___ = state
    w = math.sqrt(s*r)
    h = s/w
    return (cx, cy, w, h)


def _track(net, meta, data, output_path):
    if isinstance(data, map):
        first_frame = next(data)
    elif isinstance(data, list):
        first_frame = data[0]
    else:
        print(f'`data` should be iterator or list, is {type(data)}')
        sys.exit(-1)

    tracks = defaultdict(list)
    track_id = -1
    # initialize tracker
    detections = _detect_people(net, meta, first_frame)
    for detection in detections:
        track_id += 1
        cx, cy, w, h = detection[2]
        tracks[track_id].append([cx, cy, w*h, w/h, 0, 0, 0])
    
    # TODO tomorrow: Hungarian algorithm for assignment
    '''
    # process remaining frames
    for frame in data[1:2]:
        bbs = map(lambda d: d[2], _detect_people(net, meta, frame))
        iter_bounds = max(len(tracks.keys()), len(bbs))
        cost_mat = np.zeros((iter_bounds, iter_bounds))
        # TODO this is going to cause trouble 
        for i in range(iter_bounds):
            for j in range(iter_bounds):
                _iou(_state_to_bb(tracks[i]), bb)
    '''


def main(gpu_ind, data_path, output_path):
    # setup GPU and load model
    dn.set_gpu(gpu_ind)
    net = dn.load_net(b"darknet/cfg/yolov3.cfg", b"darknet/yolov3.weights", 0)
    meta = dn.load_meta(b"darknet/cfg/coco.data")

    # delete old files and get data filenames
    _prep_output_path(output_path)
    data = _get_data_raw(data_path) # `data` is a generator

    #_test_detector(net, meta, data, output_path)
    _track(net, meta, data, output_path)

if __name__ == '__main__':
    p = ArgumentParser('Multi-Object Tracking for People in Video')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='out')
    args = p.parse_args()
    main(args.gpu, args.data, args.out)

