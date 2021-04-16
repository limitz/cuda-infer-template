#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>
#if USE_NVJPEG
#include <nvjpeg.h>
#else
#include <jpeglib.h>
#endif

#include <display.h>
#include <pthread.h>
#include <math.h>
#include <inference.h>
#include <operators.h>
#include <asyncwork.h>
#include <jpegcodec.h>
#include <file.h>

#ifndef TITLE
#define TITLE "CUDA INFERENCE DEMO"
#endif

#ifndef USE_NVJPEG
#define USE_NVJPEG 0
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

// RGB interleaved as 3 byte tupels
__global__
void f_jpeg(float4* out, int pitch_out, uint8_t* rgb, int width, int height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= 900 || y >= 900) return;

	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			rgb[0 + y * width * 3 + x * 3] / 255.0f,
			rgb[1 + y * width * 3 + x * 3] / 255.0f,
			rgb[2 + y * width * 3 + x * 3] / 255.0f,
			1);
}
__global__
void f_normalize(float* normalized, uint8_t* rgb, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= 900 || y >= 900) return;
	size_t scstride = 300 * 300;
	size_t offset = y * width + x;
	size_t soffset = (y / 3) * (width/3) + x / 3;

	normalized[soffset + 0 * scstride] = rgb[offset*3 + 2] - 104.0f; 
	normalized[soffset + 1 * scstride] = rgb[offset*3 + 1] - 117.0f; 
	normalized[soffset + 2 * scstride] = rgb[offset*3 + 0] - 123.0f; 
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
		case 0x80:
		case 0x86:
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

int main(int /*argc*/, char** /*argv*/)
{
	int rc;
	cudaStream_t stream = 0;

	try 
	{
		printf("Selecting the best GPU\n");
		selectGPU();
		
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";

		const char* jpegPath = "showroom.jpg";
		printf("Loading \"%s\"\n", jpegPath);
		
		JpegCodec codec;
		codec.prepare(WIDTH, HEIGHT, 3);
		{
			cudaMalloc(&imageBuffer, WIDTH * HEIGHT * 3);
			
			File jpeg;
			jpeg.readAll(jpegPath);
#if USE_NVJPEG
			codec.decodeToDeviceMemoryGPU(imageBuffer, jpeg.data(), jpeg.size(), stream);
#else
			codec.decodeToDeviceMemoryCPU(imageBuffer, jpeg.data(), jpeg.size(), stream);
#endif
			cudaStreamSynchronize(stream);
		}
	
		// copy to output folder
		const char* modelPath = "../../models/nvidia_ssd.960x540.engine";
		printf("Loading \"%s\"\n", modelPath);
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
			/*
			f_segment<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				(int*)model.outputFrame.data,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);
			*/
			int32_t keepCount = -1;
			cudaMemcpyAsync(&keepCount, model.keepCount.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
			
			float boxes[7 * 200];
			cudaMemcpyAsync(&boxes, model.boxesFrame.data, model.boxesFrame.length, cudaMemcpyDeviceToHost, stream);

			const char* classes[] = { "background", "plane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
				"cat","chair", "cow", "table", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", 
				"sofa","train", "tv" };

			// copies the CUDA.frame.data to GL.pbaddr
			// and unmaps the GL.pbo
			display.cudaFinish(stream);
			display.render(stream);
		
			cudaStreamSynchronize(stream);
			for (int i=0; i<keepCount; i++)
			{
				float* detection = boxes + i * 7;
				const char* className = classes[(int)(detection[1])];
				float confidence = detection[2];
				float xmin = detection[3] * 900;
				float ymin = detection[4] * 900;
				float xmax = detection[5] * 900;
				float ymax = detection[6] * 900;

				if (confidence < 0.1) continue;
				printf("%-15s %0.02f (%f,%f,%f,%f)\n", className, confidence, xmin, ymin, xmax, ymax);
			}
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
