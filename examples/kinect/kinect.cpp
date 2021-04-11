#include <kinect.h>

Kinect::Kinect()
{
	_device = nullptr;
	_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;

	_config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
	_config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
	_config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
	_config.synchronized_images_only = true;
}

Kinect::~Kinect()
{
	
}
	
void Kinect::setColorResolution(size_t res)
{
	switch (res)
	{
	case  720: _config.color_resolution = K4A_COLOR_RESOLUTION_720P;  break;
	case 1080: _config.color_resolution = K4A_COLOR_RESOLUTION_1080P; break;
	case 1440: _config.color_resolution = K4A_COLOR_RESOLUTION_1440P; break;
	case 1536: _config.color_resolution = K4A_COLOR_RESOLUTION_1536P; break;
	case 2160: _config.color_resolution = K4A_COLOR_RESOLUTION_2160P; break;
	case 3072: _config.color_resolution = K4A_COLOR_RESOLUTION_3072P; break;
	default:
		   throw "Resolution not supported";
	}
}

void Kinect::setDepthMode(bool nfov, bool binned)
{
	_config.depth_mode = nfov ? (binned ? K4A_DEPTH_MODE_NFOV_2X2BINNED : K4A_DEPTH_MODE_NFOV_UNBINNED)
	                          : (binned ? K4A_DEPTH_MODE_WFOV_2X2BINNED : K4A_DEPTH_MODE_WFOV_UNBINNED);

	
}
void Kinect::setFramesPerSecond(size_t rate)
{
	switch(rate)
	{
	case  5: _config.camera_fps = K4A_FRAMES_PER_SECOND_5; break;
	case 15: _config.camera_fps = K4A_FRAMES_PER_SECOND_15; break;
	case 30: _config.camera_fps = K4A_FRAMES_PER_SECOND_30; break;
	default: throw "Frames/sec not supported";
	}
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
	k4a_device_stop_cameras(_device);
}


Kinect::Capture* Kinect::capture()
{
	if (!_device) return nullptr;
	k4a_capture_t capture;
	int rc = k4a_device_get_capture(_device, &capture, 0);
	switch (rc)
	{
		case K4A_WAIT_RESULT_FAILED: throw "Unable to capture kinect";
		case K4A_WAIT_RESULT_TIMEOUT: return nullptr;
		case K4A_WAIT_RESULT_SUCCEEDED: break;
	}

	return new Capture(capture, _transformation);
}

Kinect::Capture::Capture(k4a_capture_t capture, k4a_transformation_t transformation)
{
	_capture = capture;

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

	ir.image = nullptr;
	
	int rc = k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
			color.width, color.height, color.width * sizeof(uint16_t),
			&transformedDepth.image);
	if (K4A_RESULT_SUCCEEDED != rc) throw "Unable to create transformed depth image";

	rc = k4a_transformation_depth_image_to_color_camera(
			transformation, 
			depth.image, 
			transformedDepth.image);
	transformedDepth.data = k4a_image_get_buffer(transformedDepth.image);
	transformedDepth.size = k4a_image_get_size(transformedDepth.image);
	transformedDepth.width = k4a_image_get_width_pixels(transformedDepth.image);
	transformedDepth.height = k4a_image_get_height_pixels(transformedDepth.image);
}

Kinect::Capture::~Capture()
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

	if (ir.image)
	{
		k4a_image_release(ir.image);
		color.image = nullptr;
	}
	if (transformedDepth.image)
	{
		k4a_image_release(transformedDepth.image);
		color.image = nullptr;
	}
	if (_capture)
	{
		k4a_capture_release(_capture);
		_capture = nullptr;
	}
}
