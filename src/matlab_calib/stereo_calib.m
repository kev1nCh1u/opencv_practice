% Auto-generated by stereoCalibrator app on 23-Sep-2021
%-------------------------------------------------------


% Define images to process
imageFileNames1 = {'/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/01.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/02.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/03.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/04.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/05.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/06.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/07.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/08.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/09.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/10.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/11.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/12.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/13.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/14.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/15.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/16.jpg',...
    };
imageFileNames2 = {'/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/01.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/02.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/03.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/04.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/05.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/06.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/07.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/08.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/09.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/10.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/11.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/12.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/13.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/14.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/15.jpg',...
    '/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/16.jpg',...
    };

% Detect checkerboards in images
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames1, imageFileNames2);

% Generate world coordinates of the checkerboard keypoints
squareSize = 27;  % in units of 'millimeters'
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

% Read one of the images from the first stereo pair
I1 = imread(imageFileNames1{1});
[mrows, ncols, ~] = size(I1);

% Calibrate the camera
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(stereoParams);

% Visualize pattern locations
h2=figure; showExtrinsics(stereoParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, stereoParams);

% You can use the calibration data to rectify stereo images.
I2 = imread(imageFileNames2{1});
[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);
cat_j = cat(2,J1,J2);
figure; imshow(cat_j);

cat_j_size = size(cat_j);
gap = 27;
for i = gap:gap:cat_j_size(1)
    figure(3);line([0 cat_j_size(2),], [i i]);
end

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('StereoCalibrationAndSceneReconstructionExample')
% showdemo('DepthEstimationFromStereoVideoExample')