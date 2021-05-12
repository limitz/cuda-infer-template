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

	void* buffer() const { return _buffer; }

	void prepare(int width, int height, int channels, int quality);
	void unprepare();
	void decodeToDeviceMemoryCPU(void* dst, const void* src, size_t size, cudaStream_t stream);
	void decodeToDeviceMemoryGPU(void* dst, const void* src, size_t size, cudaStream_t stream);
	void encodeToHostMemoryGPU(void* dst, const void* src, size_t *size, cudaStream_t stream);
	void encodeToHostMemoryCPU(void* dst, const void* src, size_t *size, cudaStream_t stream);
	void encodeCPU(void* dst, size_t *size);

private:
	struct jpeg_decompress_struct _dinfo;
	struct jpeg_compress_struct _cinfo;
	struct jpeg_error_mgr _djerr;
	struct jpeg_error_mgr _cjerr;
	size_t _width, _height, _channels;
	uint8_t* _buffer;
	JSAMPARRAY _scanlines;
};
