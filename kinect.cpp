#include <kinect.h>
#if USE_KINECT

Kinect::Kinect()
{
	_device = nullptr;
	_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

	_config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
	_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
	_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
	_config.synchronized_images_only = true;

	depth.image = nullptr;
	color.image = nullptr;
	_capture = nullptr;
}

Kinect::~Kinect()
{
	
}

void Kinect::open()
{
	int rc;

	rc = k4a_device_get_installed_count();
	if (rc <= 0) throw "No kinect found";

	rc = k4a_device_open(K4A_DEVICE_DEFAULT, &_device);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to open kinect";
}

void Kinect::close()
{
}

void Kinect::start()
{
	int rc;
	rc = k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
			1920, 1080, 1920 * sizeof(uint16_t),
			&transformedDepth.image);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to create transformed depth image";
	
	rc = k4a_device_start_cameras(_device, &_config);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to start cameras";

	rc = k4a_device_get_calibration(_device, _config.depth_mode, _config.color_resolution, &_calibration);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to calibrate device";

	rc = k4a_device_set_color_control(_device, 
			K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE, 
			K4A_COLOR_CONTROL_MODE_AUTO, 0);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to set color control";

	_transformation = k4a_transformation_create(&_calibration);

}

void Kinect::stop()
{
	if (depth.image) 
	{
		k4a_image_release(depth.image);
		depth.image = nullptr;
	}

	if (color.image)
	{
		k4a_image_release(color.image);
		color.image = nullptr;
	}

	if (_capture)
	{
		k4a_capture_release(_capture);
		_capture = nullptr;
	}
	k4a_device_stop_cameras(_device);
	k4a_image_release(transformedDepth.image);
	transformedDepth.image = nullptr;
}


bool Kinect::capture()
{
	if (depth.image) 
	{
		k4a_image_release(depth.image);
		depth.image = nullptr;
	}

	if (color.image)
	{
		k4a_image_release(color.image);
		color.image = nullptr;
	}

	if (_capture)
	{
		k4a_capture_release(_capture);
		_capture = nullptr;
	}
	
	int rc = k4a_device_get_capture(_device, &_capture, 0);
	switch (rc)
	{
		case K4A_WAIT_RESULT_FAILED: throw "Unable to capture kinect";
		case K4A_WAIT_RESULT_TIMEOUT: return false;
		case K4A_WAIT_RESULT_SUCCEEDED: break;
	}

	depth.image = k4a_capture_get_depth_image(_capture);
	depth.data = k4a_image_get_buffer(depth.image);
	depth.size = k4a_image_get_size(depth.image);
	depth.width = k4a_image_get_width_pixels(depth.image);
	depth.height = k4a_image_get_height_pixels(depth.image);

	color.image = k4a_capture_get_color_image(_capture);
	color.data = k4a_image_get_buffer(color.image);
	color.size = k4a_image_get_size(color.image);
	color.width = k4a_image_get_width_pixels(color.image);
	color.height = k4a_image_get_height_pixels(color.image);

	return true;
}
#endif
