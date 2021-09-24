I1 = imread('/home/kevin/src/opencv_practice/img/stereo_calibration/new/1/01.jpg');
I2 = imread('/home/kevin/src/opencv_practice/img/stereo_calibration/new/2/01.jpg');

[J1, J2] = rectifyStereoImages(I1, I2, stereoParams);
subplot(1, 2, 1), imshow(J1)
subplot(1, 2, 2), imshow(J2)

% [J1, J2] = rectifyStereoImages(I1, I2, stereoParams, 'OutputView','full');
% imshow(stereoAnaglyph(J1,J2))
