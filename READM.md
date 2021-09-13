# opencv practice


## dir
1:left 2:right


## install
    pip3 install opencv-contrib-python

## circle
    python ./src/color/detect_circles.py -i img/orange.jpg


## opencv calibration sample
    ./build/camera_calibration in_VID5.xml

    ./build/stereo_calib  -w=9 -h=6 -s=27 img/stereo_calibration/stereo_calib.xml

    ./build/stereo_match img/stereo_calibration/left01.jpg img/stereo_calibration/right01.jpg --algorithm=sgbm --max-disparity=32 --blocksize=1 -i=intrinsics.yml -e=extrinsics.yml