# Multi-Object Tracker
A simple tracker based on the [SORT](https://arxiv.org/pdf/1602.00763v2.pdf)
paper. Uses YOLOv3 as the detector, the Hungarian algorithm to solve the data
association problem, and a simple constant velocity motion model Kalman filter.

## Setup
Install [darknet](https://pjreddie.com/darknet/) in the project root. Also, put
data frames (numbered) under a folder in the project root. You'll need to
modify paths in `darknet/python/darknet.py` as appropriate.

Use `python tracker.py --help` to see what relevant directories should be named

### Python dependencies
The usual: `scipy`, `numpy`, `opencv`(cv2)

## TODO
Keeping this log to help me organize work between sessions. Currently two frame
tracking and association works w/o using the KF. Need to integrate this. Also,
new tracks work pretty flawlessly, though `MIN_IOU` is probably way too low.
Need to kill tracks after `T_MIN`. Also, need to profile and see if I can
speedup. Maybe the cython calls to darknet are too slow, I don't know anything
about cython. 
