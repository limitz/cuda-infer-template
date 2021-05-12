#include <jpegcodec.h>

JpegCodec::JpegCodec()
{
	_width = 0;
	_height = 0;
	_channels = 0;
	_buffer = nullptr;
	_scanlines = nullptr;
	
	_dinfo.err = jpeg_std_error(&_djerr);
	_cinfo.err = jpeg_std_error(&_cjerr);
}
	
JpegCodec::~JpegCodec()
{
	free(_buffer);
	free(_scanlines);
}

void JpegCodec::prepare(int width, int height, int channels, int quality)
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

	for (size_t i=0; i<_height; i++)
	{
		_scanlines[i] = (JSAMPROW) (_buffer + i * _width * _channels);
	}

	jpeg_create_decompress(&_dinfo);
	jpeg_create_compress(&_cinfo);
	
	_cinfo.image_width = _width;
	_cinfo.image_height = height;
	_cinfo.input_components = 3;
	_cinfo.in_color_space = JCS_RGB; 
	jpeg_set_defaults(&_cinfo);
	jpeg_set_quality(&_cinfo, quality, 1);
}

void JpegCodec::unprepare()
{
	jpeg_destroy_decompress(&_dinfo);
	jpeg_destroy_compress(&_cinfo);
}

void JpegCodec::encodeCPU(void* dst, size_t *size)
{
	//cudaMemcpyAsync(_buffer, src, _width * _height * _channels, cudaMemcpyDeviceToHost, stream);
	//cudaStreamSynchronize(stream);
	
	jpeg_mem_dest(&_cinfo, (uint8_t**)&dst, size);
	jpeg_start_compress(&_cinfo, 1);
	while (_cinfo.next_scanline < _cinfo.image_height)
	{
		jpeg_write_scanlines(&_cinfo, _scanlines + _cinfo.next_scanline, _cinfo.image_height - _cinfo.next_scanline);
	}
	jpeg_finish_compress(&_cinfo);
}

void JpegCodec::decodeToDeviceMemoryCPU(void* dst, const void* src, size_t size, cudaStream_t stream)
{
	jpeg_mem_src(&_dinfo, (uint8_t*)src, size);
	jpeg_read_header(&_dinfo, 1);
	jpeg_calc_output_dimensions(&_dinfo);

	if (_dinfo.output_width != _width 
	||  _dinfo.output_height != _height
	||  _dinfo.output_components != (int) _channels)
	{
		jpeg_abort_decompress(&_dinfo);
		throw "Invalid image format";
	}
	jpeg_start_decompress(&_dinfo);
	while (_dinfo.output_scanline < _dinfo.output_height)
	{
		jpeg_read_scanlines(&_dinfo, _scanlines + _dinfo.output_scanline,_dinfo.output_height - _dinfo.output_scanline);
	}
	jpeg_finish_decompress(&_dinfo);

	cudaMemcpyAsync(dst, _buffer, _width * _height * _channels, cudaMemcpyHostToDevice, stream);
}

#if USE_NVJPEG
void JpegCodec::decodeToDeviceMemoryGPU(void* dst, const void* src, size_t size, cudaStream_t stream)
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
	nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGBI;
	nvjpegJpegStateCreate(handle, &state);
	nvjpegGetImageInfo(handle, (uint8_t*) src, size, &channels, &subsampling, widths, heights);

	if (widths[0] != (int)_width
	||  heights[0] != (int)_height)
	{
		nvjpegJpegStateDestroy(state);
		nvjpegDestroy(handle);
		throw "Invalid image format";
	}

	nvjpegImage_t output;
	output.channel[0] = (uint8_t*) dst;
	output.pitch[0] = widths[0] * _channels;

	nvjpegDecode(handle, state, (uint8_t*)src, size, fmt, &output, stream);
	nvjpegJpegStateDestroy(state);
	nvjpegDestroy(handle);

}
#endif
