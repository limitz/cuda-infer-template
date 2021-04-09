#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

class File
{
public:
	File(void* data = nullptr, size_t size = 0, bool transferOwnership=true);
	~File();

	void readAll(const char* filename);
	void save(const char* filename);
	void saveCompressed(const char* filename, int compressionLevel = 6);

	size_t size() const { return _size; }

	const void*    data()  const { return _data; }
	const char*    text()  const { return (char*) _data; }
	const uint8_t* bytes() const { return (uint8_t*)_data; }

private:
	size_t _size;
	void*  _data;
	bool _mustFree;
};
