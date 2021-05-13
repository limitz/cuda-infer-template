#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>


class VideoDevice
{
public:
	VideoDevice(const char* device);
	~VideoDevice();

	class Capture
	{
	public:
		Capture(int fd, v4l2_buffer buffer, void* data, size_t width, size_t height);
		~Capture();

		size_t size;
		size_t width;
		size_t height;
		void* data;
	private:
		int _fd;
		v4l2_buffer _buffer;
	};

	void setResolution(size_t width, size_t height);
	void setFramesPerSecond(size_t rate);

	void open();
	void start();
	void stop();
	void close();

	VideoDevice::Capture* capture();

private:
	int _fd;
	char* _device;
	v4l2_format _imageFormat;
	v4l2_buffer _imageBuffer;
	v4l2_capability _capability;
	void* _buffer;
};

