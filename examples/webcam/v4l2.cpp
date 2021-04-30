#include <v4l2.h>

VideoDevice::VideoDevice(const char* device)
{
	_device = strdup(device);

	_imageFormat.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	_imageFormat.fmt.pix.width = 1280;
	_imageFormat.fmt.pix.height = 720;
	_imageFormat.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
	_imageFormat.fmt.pix.field = V4L2_FIELD_NONE;
}

VideoDevice::~VideoDevice()
{
	free(_device);
}
	
void VideoDevice::setResolution(size_t width, size_t height)
{
	_imageFormat.fmt.pix.width = width;
	_imageFormat.fmt.pix.height = height;
}

void VideoDevice::setFramesPerSecond(size_t /*rate*/)
{
}


void VideoDevice::open()
{
	int rc; 

	_fd = ::open(_device, O_RDWR);
	if (_fd < 0) throw "Unable to open video device";
	
	rc = ioctl(_fd, VIDIOC_QUERYCAP, &_capability);
	if (rc < 0) throw "Unable to query video capabilities";
	
	rc = ioctl(_fd, VIDIOC_S_FMT, &_imageFormat);
	if (rc < 0) throw "Unable to set video format";

	v4l2_requestbuffers req;
	memset(&req, 0, sizeof(req));
	req.count = 1;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	rc = ioctl(_fd, VIDIOC_REQBUFS, &req);
	if (rc < 0) throw "Unable to request video buffer";

	v4l2_buffer buffer;
	memset(&buffer, 0, sizeof(buffer));

	buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buffer.memory = V4L2_MEMORY_MMAP;
	buffer.index = 0;

	rc = ioctl(_fd, VIDIOC_QUERYBUF, &buffer);
	if (rc < 0) throw "Unable to query video buffer";

	_buffer = mmap(NULL, buffer.length, PROT_READ|PROT_WRITE, MAP_SHARED, _fd, buffer.m.offset);
	memset(_buffer, 0, buffer.length);

	memset(&_imageBuffer,0, sizeof(_imageBuffer));

	_imageBuffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	_imageBuffer.memory = V4L2_MEMORY_MMAP;
	_imageBuffer.index = 0;
}

void VideoDevice::close()
{
	::close(_fd);
}

void VideoDevice::start()
{
	int rc;
	rc = ioctl(_fd, VIDIOC_STREAMON, &_imageBuffer.type);
	if (rc < 0) throw "Unable to start video stream";
	
	rc = ioctl(_fd, VIDIOC_QBUF, &_imageBuffer);
	if (rc < 0) throw "Unable to queue video buffer";
}

void VideoDevice::stop()
{
	int rc;
	
	rc = ioctl(_fd, VIDIOC_DQBUF, &_imageBuffer);
	if (rc < 0) throw "Unable to dequeue video buffer";
	
	rc = ioctl(_fd, VIDIOC_STREAMOFF, &_imageBuffer.type);
	if (rc < 0) throw "Unable to stop video stream";
}


VideoDevice::Capture* VideoDevice::capture()
{
	return new Capture(_fd, _imageBuffer, _buffer, _imageFormat.fmt.pix.width, _imageFormat.fmt.pix.height);
}

VideoDevice::Capture::Capture(int fd, v4l2_buffer buffer, void* d, size_t w, size_t h)
{
	int rc;
	_fd = fd;
	_buffer = buffer;
	rc = ioctl(_fd, VIDIOC_DQBUF, &_buffer);
	if (rc < 0) throw "Unable to dequeue video buffer";

	data = d;
	size = _buffer.bytesused;
	width = w;
	height = h;
}

VideoDevice::Capture::~Capture()
{
	int rc = ioctl(_fd, VIDIOC_QBUF, &_buffer);
	if (rc < 0) throw "Unable to dequeue video buffer";
}
