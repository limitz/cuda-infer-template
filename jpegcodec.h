#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#if USE_NVJPEG 
#include <nvjpeg.h>
#endif
#include <jpeglib.h>

class JpegCodec
{
public:
	
	JpegCodec();
	~JpegCodec();

	void prepare(int width, int height, int channels);
	void unprepare();
	void decodeToDeviceMemoryCPU(void* dst, const void* src, int size, cudaStream_t stream);
	void decodeToDeviceMemoryGPU(void* dst, const void* src, int size, cudaStream_t stream);

private:
	struct jpeg_decompress_struct _cinfo;
	struct jpeg_error_mgr _jerr;
	size_t _width, _height, _channels;
	uint8_t* _buffer;
	JSAMPARRAY _scanlines;
};
