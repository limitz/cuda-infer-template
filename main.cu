#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <math.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <display.h>
#include <pthread.h>
#include <math.h>
#include <inference.h>
#include <operators.h>

#include <asyncwork.h>

#ifndef TITLE
#define TITLE "CUDA INFERENCE DEMO"
#endif

//width and height defines come from inference.h at the moment

static uint8_t* imageBuffer = {0};

__global__
void f_test(float4* out, int pitch_out, int width, int height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			(float) x / width, 
			(float) y / height, 
			0, 1);
}

__global__
void f_jpeg(float4* out, int pitch_out, uint8_t* rgb, int width, int height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	int cstride = width * height;
	if (x >= width || y >= height) return;

	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			rgb[0 * cstride + y * width + x] / 255.0f,
			rgb[1 * cstride + y * width + x] / 255.0f,
			rgb[2 * cstride + y * width + x] / 255.0f,
			1);
}
__global__
void f_normalize(float* normalized, uint8_t* rgb, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	size_t cstride = width * height;
	size_t scstride = (width/SCALE) * (height/SCALE);
	size_t offset = y * width + x;
	size_t soffset = (y / SCALE) * (width/SCALE) + x / SCALE;

	normalized[soffset + 0 * scstride] = (rgb[offset + 0 * cstride]/255.0f - 0.485f) / (0.229f); 
	normalized[soffset + 1 * scstride] = (rgb[offset + 1 * cstride]/255.0f - 0.456f) / (0.224f); 
	normalized[soffset + 2 * scstride] = (rgb[offset + 2 * cstride]/255.0f - 0.406f) / (0.225f); 
}
__global__
void f_segment(float4* out, int pitch_out, int* seg, int width, int height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	float alpha = 0.4;
	int classification = seg[(y/SCALE) * (width/SCALE) + (x/SCALE)];
	float4 color = classification ? make_float4(
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.00f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.33f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.66f) * 2 * M_PI),
			alpha) : make_float4(0,0,0,0);

	int idx = y * pitch_out/sizeof(float4) + x;
	out[idx] = out[idx] * (1-color.w) + color;
}

int smToCores(int major, int minor)
{
	switch ((major << 4) | minor)
	{
		case (9999 << 4 | 9999):
			return 1;
		case 0x30:
		case 0x32:
		case 0x35:
		case 0x37:
			return 192;
		case 0x50:
		case 0x52:
		case 0x53:
			return 128;
		case 0x60:
			return 64;
		case 0x61:
		case 0x62:
			return 128;
		case 0x70:
		case 0x72:
		case 0x75:
			return 64;
		default:
			return 0;
	};
}

void selectGPU()
{
	int rc;
	int maxId = -1;
	uint16_t maxScore = 0;
	int count = 0;
	cudaDeviceProp prop;

	rc = cudaGetDeviceCount(&count);
	if (cudaSuccess != rc) throw "cudaGetDeviceCount error";
	if (count == 0) throw "No suitable cuda device found";

	for (int id = 0; id < count; id++)
	{
		rc = cudaGetDeviceProperties(&prop, id);
		if (cudaSuccess != rc) throw "Unable to get device properties";
		if (prop.computeMode == cudaComputeModeProhibited) 
		{
			printf("GPU %d: PROHIBITED\n", id);
			continue;
		}
		int sm_per_multiproc = smToCores(prop.major, prop.minor);
		
		printf("GPU %d: \"%s\"\n", id, prop.name);
		printf(" - Compute capability: %d.%d\n", prop.major, prop.minor);
		printf(" - Multiprocessors:    %d\n", prop.multiProcessorCount);
		printf(" - SMs per processor:  %d\n", sm_per_multiproc);
		printf(" - Clock rate:         %d\n", prop.clockRate);

		uint64_t score =(uint64_t) prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;
		if (score > maxScore) 
		{
			maxId = id;
			maxScore = score;
		}
	}

	if (maxId < 0) throw "All cuda devices prohibited";

	rc = cudaSetDevice(maxId);
	if (cudaSuccess != rc) throw "Unable to set cuda device";

	rc = cudaGetDeviceProperties(&prop, maxId);
	if (cudaSuccess != rc) throw "Unable to get device properties";

	printf("\nSelected GPU %d: \"%s\" with compute capability %d.%d\n\n", 
		maxId, prop.name, prop.major, prop.minor);
}


void loadJpeg(const char* path, cudaStream_t stream)
{
	int rc;
	FILE* f = fopen(path, "rb");
	if (!f) throw "Unable to open JPEG";

	size_t read = 0;
	size_t allocated = 0;
	size_t increment = 4096;
	uint8_t* data = 0;

	while (!feof(f))
	{
		if (read <= allocated)
		{
			allocated += increment;
			void* tmp = realloc(data, allocated);
			if (!tmp) 
			{
				free(data);
				throw "Unable to allocate JPEG memory";
			}
			data = (uint8_t*) tmp;
		}
		int r = fread(data + read, 1, allocated - read, f);
		if (ferror(f)) 
		{
			printf("ERROR = %d\n", ferror(f));
			free(data);
			throw "Error reading file";
		}
		read += r;
	}

	nvjpegHandle_t handle;
	rc = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, NULL, NULL, 0, &handle);
	if (cudaSuccess != rc) throw "Unable to create jpeg backend";

	int channels;
	int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
	nvjpegChromaSubsampling_t subsampling;
	nvjpegJpegState_t state;
	nvjpegOutputFormat_t fmt = NVJPEG_OUTPUT_RGB;
	nvjpegJpegStateCreate(handle, &state);
	//nvjpegDecodeBatchedInitialize(handle, state, 1, 1, fmt);
	nvjpegGetImageInfo(handle, data, read, &channels, &subsampling, widths, heights);
	cudaStreamSynchronize(stream);
	
	printf("Got image of size %lu,  info %d %d %d\n", read, channels, widths[0], heights[0]);

	// to RGB (3 channels, not interleaved, tightly packed for inference)
	channels = 3;
	size_t channelSize = widths[0] * heights[0];
	nvjpegImage_t output;
	
	rc = cudaMalloc(&imageBuffer, channels * channelSize);
	if (cudaSuccess != rc) throw "Unable to allocate image buffer on device";

	/*
	rc = cudaMalloc(&imageBufferNormalized, channels * channelSize * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate image buffer on device";
	*/
	for (int i=0; i<channels; i++)
	{
		output.channel[i] = &imageBuffer[i * channelSize];
		output.pitch[i] = widths[0];
	}

	// not pipelined at the moment
	cudaStreamSynchronize(stream);
	rc = cudaGetLastError();
	if (cudaSuccess != rc) throw "Unable to prepare jpeg";

	printf("Decoding\n");
	nvjpegDecode(handle, state, data, read, fmt, &output, stream);
	cudaStreamSynchronize(stream);
	rc = cudaGetLastError();
	if (cudaSuccess != rc) throw "Unable to decode jpeg";
	
	printf("Decoded\n");
	nvjpegJpegStateDestroy(state);
	nvjpegDestroy(handle);
	cudaStreamSynchronize(stream);
	rc = cudaGetLastError();
	if (cudaSuccess != rc) throw "Unable to destroy jpeg resources";
}

int main(int /*argc*/, char** /*argv*/)
{
	int rc;
	cudaStream_t stream = 0;

	printf("Selecting the best GPU\n");
	selectGPU();

	try 
	{
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";


		const char* jpegPath = "sheep.jpg";
		printf("Loading \"%s\"\n", jpegPath);
		loadJpeg(jpegPath, stream);
		cudaDeviceSynchronize();
	
		// copy to output folder
		const char* modelPath = "models/fcn_resnet101.960x540.engine";
		printf("Loading \"%s\"", modelPath);
		Model model(modelPath);

		printf("Creating screen\n");
		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		cudaDeviceSynchronize();
		
		dim3 blockSize = { 16, 16 };
		dim3 gridSize = { 
			(WIDTH  + blockSize.x - 1) / blockSize.x, 
			(HEIGHT + blockSize.y - 1) / blockSize.y 
		}; 

		display.cudaMap(stream);
		while (true)
		{	
			f_test<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);

			f_normalize<<<gridSize, blockSize, 0, stream>>>(
				(float*)model.inputFrame.data,
				imageBuffer,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);
			f_jpeg<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				imageBuffer,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);
			
			model.infer(stream);

			f_segment<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				(int*)model.outputFrame.data,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);

			// copies the CUDA.frame.data to GL.pbaddr
			// and unmaps the GL.pbo
			display.cudaFinish(stream);
			display.render(stream);
			
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			// check escape pressed
			if (display.events()) 
			{
				display.cudaUnmap(stream);
				cudaStreamDestroy(stream);
				return 0;
			}
			usleep(1000);
		}
	}
	catch (const char* &ex)
	{
		fprintf(stderr, "ERROR: %s\n", ex);
		fflush(stderr);
	 	return 1;
	}
	return 0;
}
