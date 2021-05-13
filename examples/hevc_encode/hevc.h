#pragma once
#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <unistd.h>
#include <stdint.h>

#include <nvEncodeAPI.h>

class HEVCEncoder
{
public:
	HEVCEncoder(int deviceOrdinal, size_t width, size_t height, size_t fps);
	~HEVCEncoder();

private:
	int getCapability(GUID id, NV_ENC_CAPS cap);
	NV_ENCODE_API_FUNCTION_LIST _nvenc;
	void* _encoder;
	CUdevice _device;
	CUcontext _context;
};
