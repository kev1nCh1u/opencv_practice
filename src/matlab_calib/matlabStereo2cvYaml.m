% matlab_stereo_param
fileName = "../../param/matlab_stereo_param.yaml";

matlab2opencv(fileName, stereoParams.CameraParameters1.IntrinsicMatrix, "IntrinsicMatrix1", "w")
matlab2opencv(fileName, stereoParams.CameraParameters1.RadialDistortion, "RadialDistortion1", "a")
matlab2opencv(fileName, stereoParams.CameraParameters1.TangentialDistortion, "TangentialDistortion1", "a")

matlab2opencv(fileName, stereoParams.CameraParameters2.IntrinsicMatrix, "IntrinsicMatrix2", "a")
matlab2opencv(fileName, stereoParams.CameraParameters2.RadialDistortion, "RadialDistortion2", "a")
matlab2opencv(fileName, stereoParams.CameraParameters2.TangentialDistortion, "TangentialDistortion2", "a")

matlab2opencv(fileName, stereoParams.CameraParameters1.ImageSize, "ImageSize", "a")
matlab2opencv(fileName, stereoParams.RotationOfCamera2, "RotationOfCamera2", "a")
matlab2opencv(fileName, stereoParams.TranslationOfCamera2, "TranslationOfCamera2", "a")

