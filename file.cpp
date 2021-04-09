#include <file.h>
#include <lz4frame.h>


File::File(void* data, size_t size, bool transferOwnership) 
{ 
	_size = size; 
	_data = data; 
	_mustFree = transferOwnership;
}

File::~File() 
{ 
	if (_data && _mustFree) free(_data); 
}

void File::readAll(const char* filename)
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
				fclose(f);
				free(_data);
				throw "Unable to allocated enough data for file";
			}
			_data = tmp;
		}
		int r = fread(((uint8_t*)_data) + _size, 1, allocated - _size, f);
		if (ferror(f))
		{
			fprintf(stderr, "Error reading file (%08X)", ferror(f));
			fclose(f);
			free(_data);
			throw "Unable to read file";
		}
		_size += r;
	}
	fclose(f);
}

void File::save(const char* filename)
{
	FILE* f = fopen(filename, "wb");
	if (!f) throw "Unable to open file for writing";

	size_t written = 0;
	while (written < _size)
	{
		int rc = fwrite(bytes() + written, 1, _size - written, f);
		if (rc <= 0)
		{
			fclose(f);
			throw "Unable to save file";
		}
		written += rc;
	}
	fclose(f);
}

void File::saveCompressed(const char* filename, int compressionLevel)
{
	FILE* f = fopen(filename, "wb");
	if (!f) throw "Unable to open file for writing";
		
	LZ4F_compressionContext_t ctx;
	size_t chunk = 1024*1024;
	size_t ccc = LZ4F_createCompressionContext(&ctx, LZ4F_VERSION);
	if (LZ4F_isError(ccc)) 
	{
		fclose(f);
		throw "Unable to create compression context";
	}

	LZ4F_preferences_t prefs = {
		.frameInfo = {
			.blockSizeID = LZ4F_max1MB,
			.blockMode = LZ4F_blockLinked,
			.contentChecksumFlag = LZ4F_noContentChecksum,
			.frameType = LZ4F_frame,
			.contentSize = 0,
			.reserved = {0},
		},
		.compressionLevel = compressionLevel,
		.autoFlush = 0,
		.reserved = {0},
	};

	size_t out_capacity = LZ4F_compressBound(chunk, &prefs);
	uint8_t* compressed = (uint8_t*)malloc(out_capacity);
	if (!compressed) 
	{
		fclose(f);
		throw "Unable to allocate compression buffer";
	}

	size_t csize = LZ4F_compressBegin(ctx, compressed, out_capacity, &prefs);
	printf("csize = %d\n", csize);
	if (LZ4F_isError(csize)) 
	{
		free(compressed);
		fclose(f);
		throw "Unable to start compression";
	}

	size_t written = 0;
	while (written < csize)
	{
		int rc = fwrite(compressed + written, 1, csize - written, f);
		if (rc <= 0)
		{
			free(compressed);
			fclose(f);
			throw "Unable to write lz4f header";
		}
		written += rc;
	}
	for (size_t i=0; i<_size; i += chunk)
	{
		size_t remaining = _size - i;
		size_t read = remaining < chunk ? remaining : chunk;
		const uint8_t* in_ptr = bytes() + i;
		size_t incr = LZ4F_compressUpdate(ctx, compressed, out_capacity, in_ptr, read, NULL);
		printf("incr = %d\n", incr);
		if (LZ4F_isError(incr))
		{
			free(compressed);
			fclose(f);
			throw "Unable to compress file";
		}
		
		written = 0;
		while (written < incr)
		{
			int rc = fwrite(compressed + written, 1, incr - written, f);
			if (rc <= 0)
			{
				free(compressed);
				fclose(f);
				throw "Unable to write compressed data";
			}
			written += rc;
		}
	}
	size_t end = LZ4F_compressEnd(ctx, compressed, out_capacity, NULL);
	printf("end = %d\n", end);
	if (LZ4F_isError(end))
	{
		free(compressed);
		fclose(f);
		throw "Unable to compress end of file";
	}
	written = 0;
	while (written < end)
	{
		int rc = fwrite(compressed + written, 1, end - written, f);
		if (rc <= 0)
		{
			free(compressed);
			fclose(f);
			throw "Unable to write end of file";
		}
		written += rc;
	}
	free(compressed);
	fclose(f);
}
