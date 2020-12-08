from argparse import ArgumentParser
import sys, os

import cv2
sys.path.append(os.path.join(os.getcwd(),'darknet/python/'))
import darknet as dn


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
        return detection[0] == b'person'

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


def main(gpu_ind, data_path, output_path):
    # setup GPU and load model
    dn.set_gpu(gpu_ind)
    net = dn.load_net(b"darknet/cfg/yolov3.cfg", b"darknet/yolov3.weights", 0)
    meta = dn.load_meta(b"darknet/cfg/coco.data")

    # delete old files and get data filenames
    _prep_output_path(output_path)
    data = _get_data_raw(data_path) # `data` is a generator

    _test_detector(net, meta, data, output_path)

if __name__ == '__main__':
    p = ArgumentParser('Multi-Object Tracking for People in Vide')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--data', default='data')
    p.add_argument('--out', default='out')
    args = p.parse_args()
    main(args.gpu, args.data, args.out)

