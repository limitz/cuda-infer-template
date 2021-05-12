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

	struct Settings
	{

		struct
		{
			struct 
			{
				size_t width = 1024;
				size_t height = 1024;
				bool transform = false;
				bool compress = false;
			} depth;
			
			struct
			{
				size_t width = 2048;
				size_t height = 1536;
				bool transform = false;
			} color;
			int frameRate = 15;
		
		} acquisition;

		struct 
		{
			int gain = 0;
			int saturation = 128;
			int brightness = 128;
			int exposure = -1;
			int sharpness = 0;
			int contrast = 128 ;
			int whitebalance = -1;
			int backlightCompensation = 0;
			int powerlineFrequency = 1;
		} colorControl;
	};

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

	Settings settings;
	void updateSettings(bool force = false);

	void start();
	void stop();

	Kinect::Capture* capture();

private:
	Settings activeSettings;
	k4a_transformation_t _transformation;
	k4a_calibration_t _calibration;
	k4a_device_t _device;
	k4a_device_configuration_t _config;
};

