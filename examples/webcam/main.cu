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
#include <v4l2.h>
#include <config.h>

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
void f_jpeg(float4* out, int pitch_out, uint8_t* rgb, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;

	int offset = (y * width + x) * 3;
	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			rgb[0 + offset] / 255.0f,
			rgb[1 + offset] / 255.0f,
			rgb[2 + offset] / 255.0f,
			1);
}

__global__
void f_normalize(float* normalized, uint8_t* rgb, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= 300 || y >= 300) return;

	int sx = (int) (x * width / 300.f);
	int sy = (int) (y * width / 300.f);
	if (sx >= width || sy >= height) return;

	size_t offset = (sy * width + sx) * 3;
	size_t soffset = y * 300 + x;

	normalized[soffset +      0] = rgb[offset + 2] - 104.0f; 
	normalized[soffset +  90000] = rgb[offset + 1] - 117.0f; 
	normalized[soffset + 180000] = rgb[offset + 0] - 123.0f; 
}
#define OVERLAY_BOXES
__global__
void f_bbox(float4* out, int pitch_out, float* boxes, uint32_t* nboxes, size_t width, size_t height)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= width || y >= height) return;
	int idx = y * pitch_out/sizeof(float4) + x;

	float classification = 0;
	for (int i=0; i<*nboxes; i++)
	{
		float* box = boxes + i * 7;
		if (box[1] != 7 && box[1] != 15) continue;
		if (box[2] < 0.6f || box[2] > 2.0) continue;
		float minx = box[3] * width;
		float miny = box[4] * width;
		float maxx = box[5] * width;
		float maxy = box[6] * width;
		if (x < minx || x > maxx || y < miny || y > maxy) continue;
		classification = box[1] / 20.0f;
#ifdef OVERLAY_BOXES
		float4 color = classification ? make_float4(
			0.2 + 0.2 * __sinf((classification+0.00f) * 2 * M_PI),
			0.2 + 0.2 * __sinf((classification+0.33f) * 2 * M_PI),
			0.2 + 0.2 * __sinf((classification+0.66f) * 2 * M_PI),
			0.2) : make_float4(0,0,0,0);

		out[idx] = out[idx] * (1-color.w) + color;
	}
#else
		break;
	}
	float4 color = classification ? make_float4(
			0.2 + 0.2 * __sinf((classification+0.00f) * 2 * M_PI),
			0.2 + 0.2 * __sinf((classification+0.33f) * 2 * M_PI),
			0.2 + 0.2 * __sinf((classification+0.66f) * 2 * M_PI),
			0.2) : make_float4(0,0,0,0);

	out[idx] = out[idx] * (1-color.w) + color;
#endif
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
		Config config;
		config.loadFile("defaults.config");
		config.printHelp();
		config.print();

		printf("Selecting the best GPU\n");
		selectGPU();
		
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";

		size_t cam_width = config.get("WebcamWidth").uint32();
		size_t cam_height = config.get("WebcamHeight").uint32();

		JpegCodec codec;
		codec.prepare(cam_width, cam_height, 3);
		rc = cudaMalloc(&imageBuffer, cam_width * cam_height * 3);
		if (cudaSuccess != rc) throw "Unable to allocate image buffer";

		// copy to output folder
		const char* modelPath = "../../models/ssd.engine";
		const char* prototxt  = "../../models/ssd.prototxt";
		const char* caffemodel= "../../models/ssd.caffemodel";

		printf("Loading \"%s\"\n", modelPath);
		Model model(modelPath, prototxt, caffemodel);

		printf("Creating screen\n");
		CudaDisplay display(TITLE, cam_width, cam_height); 
		cudaDeviceSynchronize();
	
		VideoDevice webcam(config.get("WebcamDevice").string());
		webcam.setFramesPerSecond(config.get("WebcamFramesPerSecond").uint32());
		webcam.setResolution(cam_width, cam_height); //must be WIDTH
		webcam.open();
		webcam.start();

		dim3 blockSize = { 16, 16 };
		dim3 gridSize = {0};
		gridSize.x = (cam_width  + blockSize.x - 1) / blockSize.x;
		gridSize.y = (cam_height + blockSize.y - 1) / blockSize.y;
		

		dim3 gridSize300 = {
			(300 + blockSize.x - 1) / blockSize.x,
			(300 + blockSize.y - 1) / blockSize.y
		};
		display.cudaMap(stream);
		while (true)
		{
			auto capture = webcam.capture();
			if (capture)
			{
				
				cudaEvent_t start, stop;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, stream);
#if USE_NVJPEG
				codec.decodeToDeviceMemoryGPU(
#else
				codec.decodeToDeviceMemoryCPU(
#endif
					imageBuffer,
					capture->data,
					capture->size,
					stream);
			
				f_normalize<<<gridSize300, blockSize, 0, stream>>>(
					(float*)model.inputFrame.data,
					imageBuffer,
					cam_width,
					cam_height
			
				);
			
				f_jpeg<<<gridSize, blockSize, 0, stream>>>(
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					imageBuffer,
					cam_width,
					cam_height
				);
			
				model.infer(stream);
			
				f_bbox<<<gridSize, blockSize, 0, stream>>>(
					display.CUDA.frame.data,
					display.CUDA.frame.pitch,
					(float*) model.boxesFrame.data,
					(uint32_t*) model.keepCount.data,
					cam_width,
					cam_height
				);
			
				cudaEventRecord(stop, stream);
				cudaEventSynchronize(stop);

				float ms;
				cudaEventElapsedTime(&ms, start, stop);
				printf("CUDA elapsed time: %0.2f ms\n", ms);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);

				delete capture;
			
				display.cudaFinish(stream);
				display.render(stream);
			}
			// copies the CUDA.frame.data to GL.pbaddr
			// and unmaps the GL.pbo
		
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			// check escape pressed
			if (display.events()) 
			{
				webcam.stop();
				webcam.close();
				display.cudaUnmap(stream);
				cudaStreamSynchronize(stream);
				cudaStreamDestroy(stream);
				return 0;
			}
			usleep(100);
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
