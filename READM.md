# opencv practice


## dir
1:left 2:right


## install
    pip3 install opencv-contrib-python

## circle
    python ./src/color/detect_circles.py -i img/orange.jpg


## opencv calibration sample
    ./build/camera_calibration param/in_VID5.xml

    ./build/stereo_calib  -w=9 -h=6 -s=27 img/stereo_calibration/new/stereo_calib.xml

    ./build/stereo_match img/stereo_calibration/new/1/01.jpg img/stereo_calibration/new/2/01.jpg --algorithm=sgbm --max-disparity=32 --blocksize=1 -i=param/intrinsics.yml -e=param/extrinsics.yml

## single camera point record
    point_path.py
    combine_data.py

## parameter

f: stereoParams.CameraParameters1.FocalLength  
dis: stereoParams.TranslationOfCamera2  