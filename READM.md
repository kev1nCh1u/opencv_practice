# opencv practice

## install
    pip3 install opencv-contrib-python

## circle
    python .\src\color\detect_circles.py -i img\orange.jpg


## opencv calibration sample
    ./build/camera_calibration in_VID5.xml
    ./build/stereo_calib  -w=9 -h=6 -s=27 img/stereo_calibration/stereo_calib.xml