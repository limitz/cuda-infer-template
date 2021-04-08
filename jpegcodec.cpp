#include <jpegcodec.h>

JpegCodec::JpegCodec()
{
	_width = 0;
	_height = 0;
	_channels = 0;
	_buffer = nullptr;
	_scanlines = nullptr;
	
	_cinfo.err = jpeg_std_error(&_jerr);
}
	
JpegCodec::~JpegCodec()
{
	free(_buffer);
	free(_scanlines);
}

void JpegCodec::prepare(int width, int height, int channels)
{
	if (channels != 3) throw "Not implemented channels != 3";

	_width = width;
	_height = height;
	_channels = channels;

	_buffer = (uint8_t*) malloc(_width * _height * _channels);
	if (!_buffer) throw "Unable to allocate intermediate buffer";

	_scanlines = (JSAMPARRAY) malloc( sizeof(JSAMPROW) * height);
	if (!_scanlines)
	{
		free(_buffer);
		throw "Unable to allocate scanlines structure";
	}

	for (int i=0; i<_height; i++)
	{
		_scanlines[i] = (JSAMPROW) (_buffer + i * _width * _channels);
	}

	jpeg_create_decompress(&_cinfo);
}

void JpegCodec::unprepare()
{
	jpeg_destroy_decompress(&_cinfo);
}

void JpegCodec::decodeToDeviceMemoryCPU(void* dst, const void* src, int size, cudaStream_t stream)
{
	jpeg_mem_src(&_cinfo, (uint8_t*)src, size);
	jpeg_read_header(&_cinfo, 1);
	jpeg_calc_output_dimensions(&_cinfo);

	if (_cinfo.output_width != _width 
	||  _cinfo.output_height != _height
	||  _cinfo.output_components != _channels)
	{
		jpeg_abort_decompress(&_cinfo);
		throw "Invalid image format";
	}
	jpeg_start_decompress(&_cinfo);
	while (_cinfo.output_scanline < _cinfo.output_height)
	{
		jpeg_read_scanlines(&_cinfo, _scanlines + _cinfo.output_scanline,_cinfo.output_height - _cinfo.output_scanline);
	}
	jpeg_finish_decompress(&_cinfo);

	cudaMemcpyAsync(dst, _buffer, _width * _height * _channels, cudaMemcpyHostToDevice, stream);
}

#if USE_NVJPEG
void JpegCodec::decodeToDeviceMemoryGPU(void* dst, const void* src, int size, cudaStream_t stream)
{
	int rc;
	
	nvjpegHandle_t handle;
	rc = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, 0, &handle);
	if (cudaSuccess != rc) throw "Unable to create nvjpeg handle";

	int channels;
	int widths[NVJPEG_MAX_COMPONENT];
	int heights[NVJPEG_MAX_COMPONENT];
	nvjpegChromaSubsampling_t subsampling;
	nvjpegJpegState_t state;
	nvJpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;
	nvjpegJpegStateCreate(handle, &state);
	nvjpegGetImageInfo(handle, src, size, &channels, &subsampling, widths, heights);

	if (widths[0] != _width
	||  heights[0] != _height)
	{
		nvjpegJpegStateDestroy(state);
		nvjpegDestroy(handle);
		throw "Invalid image format";
	}

	nvjpegImage_t output;
	output.channel[0] = dst;
	output.pitch[0] = widths[0] * _channels;

	nvjpegDecode(handle, state, src, size, fmt, &output, stream);
	nvjpegJpegStateDestroy(state);
	nvjpegDestroy(handle);

}
#endif
