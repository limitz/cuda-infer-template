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

class File
{
public:
	File(void* data = nullptr, size_t size = 0) 
	{ 
		_size = size; 
		_data = data; 
	}
	~File() { if (_data) free(_data); }

	void readAll(const char* filename)
	{
		if (_data) free(_data);
		_size = 0;

		FILE* f = fopen(filename, "rb");
		if (!f) throw "Unable to open file";
		
		size_t allocated = 0;
		size_t increment = 1<<16;

		while (!feof(f))
		{
			if (_size <= allocated)
			{
				allocated += increment;
				void* tmp = realloc(_data, allocated);
				if (!tmp)
				{
					free(_data);
					throw "Unable to allocated enough data for file";
				}
				_data = tmp;
			}
			int r = fread(((uint8_t*)_data) + _size, 1, allocated - _size, f);
			if (ferror(f))
			{
				fprintf(stderr, "Error reading file (%08X)", ferror(f));
				free(_data);
				throw "Unable to read file";
			}
			_size += r;
		}
	}

	size_t size() const { return _size; }

	const void*    data()  const { return _data; }
	const char*    text()  const { return (char*) _data; }
	const uint8_t* bytes() const { return (uint8_t*)_data; }

private:
	size_t _size;
	void*  _data;
};

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
