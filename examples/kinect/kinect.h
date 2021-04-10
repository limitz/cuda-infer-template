#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <k4a/k4a.h>

class Kinect
{
public:
	Kinect();
	~Kinect();

	class Capture
	{
	public:
		Capture(k4a_capture_t capture, k4a_transformation_t transformation);
		~Capture();

		struct 
		{
			k4a_image_t image;
			size_t size;
			size_t width;
			size_t height;
			void* data;
		} depth, color, ir, transformedDepth;
	private:
		k4a_capture_t _capture;
	};

	void setColorResolution(size_t res);
	void setDepthMode(bool nfov, bool binned);
	void setFramesPerSecond(size_t rate);

	void open();
	void start();
	void stop();
	void close();

	Kinect::Capture* capture();


private:
	k4a_transformation_t _transformation;
	k4a_calibration_t _calibration;
	k4a_device_t _device;
	k4a_device_configuration_t _config;
};

