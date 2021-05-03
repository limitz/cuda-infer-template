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

#define WIDTH 1920
#define HEIGHT 1080

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

#define MIN_PROB 0.6

#ifndef USE_NVJPEG
#define USE_NVJPEG 0
#endif

//width and height defines come from inference.h at the moment

static uint8_t* imageBuffer = {0};
const char* const classNames[] = {
	"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
	"diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa","train", "tvmonitor"
};

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
void f_jpeg(float4* out, int pitch_out, uint8_t* rgb)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= WIDTH || y >= HEIGHT) return;

	out[y * pitch_out / sizeof(float4) + x] = make_float4(
			rgb[0 + y * WIDTH * 3 + x * 3] / 255.0f,
			rgb[1 + y * WIDTH * 3 + x * 3] / 255.0f,
			rgb[2 + y * WIDTH * 3 + x * 3] / 255.0f,
			1);
}

#define SDIV   (WIDTH / 300.0f)
__global__
void f_normalize(float* normalized, uint8_t* rgb)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= 300 || y >= 300) return;
	int sx = (int) (x * SDIV);
	int sy = (int) (y * SDIV);
	bool valid = sx < WIDTH && sy < HEIGHT;

	size_t offset = sy * WIDTH + sx;
	size_t scstride = 300 * 300;
	size_t soffset = y * 300 + x;

	normalized[soffset + 0 * scstride] = valid ? rgb[offset*3 + 2] - 104.0f : 0; 
	normalized[soffset + 1 * scstride] = valid ? rgb[offset*3 + 1] - 117.0f : 0; 
	normalized[soffset + 2 * scstride] = valid ? rgb[offset*3 + 0] - 123.0f : 0; 
}

__global__
void f_bbox(float4* out, int pitch_out, float* boxes, uint32_t* nboxes)
{
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);
	if (x >= WIDTH || y >= HEIGHT) return;

	int classification = 0;
	for (int i=0; i<*nboxes; i++)
	{
		float* box = boxes + i * 7;
		if (box[2] < MIN_PROB || box[2] > 1.0) continue;
		if (box[1] <= 0 || box[1] >= 21) continue;
		float minx = box[3] * WIDTH;
		float miny = box[4] * WIDTH;
		float maxx = box[5] * WIDTH;
		float maxy = box[6] * WIDTH;
		if (x < minx || x > maxx || y < miny || y > maxy) continue;
		classification = box[1];
		float alpha = 0.4;
		float4 color = classification ? make_float4(
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.00f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.33f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.66f) * 2 * M_PI),
			alpha) : make_float4(0,0,0,0);

		int idx = y * pitch_out/sizeof(float4) + x;
		out[idx] = out[idx] * (1-color.w) + color;
	}
#if 0
	float alpha = 0.4;
	float4 color = classification ? make_float4(
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.00f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.33f) * 2 * M_PI),
			alpha/2 + alpha/2 * __sinf((classification/20.0f+0.66f) * 2 * M_PI),
			alpha) : make_float4(0,0,0,0);

	int idx = y * pitch_out/sizeof(float4) + x;
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
		printf("Selecting the best GPU\n");
		selectGPU();
		
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";

		const char* jpegPath = "sheep.jpg";
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
		const char* modelPath = "../../models/ssd.engine";
		const char* prototxt  = "../../models/ssd.prototxt";
		const char* caffemodel= "../../models/ssd.caffemodel";

		printf("Loading \"%s\"\n", modelPath);
		Model model(modelPath, prototxt, caffemodel);

		printf("Creating screen\n");
		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		cudaDeviceSynchronize();
		
		dim3 blockSize = { 16, 16 };
		dim3 gridSize = { 
			(WIDTH  + blockSize.x - 1) / blockSize.x, 
			(HEIGHT + blockSize.y - 1) / blockSize.y 
		}; 

		dim3 gridSize300 = {
			(300 + blockSize.x - 1) / blockSize.x,
			(300 + blockSize.y - 1) / blockSize.y
		};
		display.cudaMap(stream);
		while (true)
		{
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
#if 0
			f_test<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);
#endif
			
			f_normalize<<<gridSize300, blockSize, 0, stream>>>(
				(float*)model.inputFrame.data,
				imageBuffer
			);
			
			f_jpeg<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				imageBuffer
			);

			cudaEventRecord(start);
			model.infer(stream);
			cudaEventRecord(stop);
			
			// Copy the bounding boxes to host memory to display them 
			// on the command line
			uint32_t count;
			cudaMemcpyAsync(&count, model.keepCount.data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);

			size_t bsize = count * 7 * sizeof(float);
			float* boxes = (float*)malloc(bsize);
			cudaMemcpyAsync(boxes, model.boxesFrame.data, bsize, cudaMemcpyDeviceToHost, stream);
	
			// Draw the boxes (from device memory)
			f_bbox<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				(float*) model.boxesFrame.data,
				(uint32_t*) model.keepCount.data);

			// This is also done by display.cudaFinish in this example.
			cudaStreamSynchronize(stream);
			
			// Here we know all drawing has been done, and all memcpy's are finished
			// Due to the synchronize above
		
			// Draw the pixelbuffer on screen
			display.cudaFinish(stream);
			display.render(stream);
		

			float ms;
			cudaEventElapsedTime(&ms, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			printf("Number of boxes: %d\n", count);
			for (size_t i=0; i<count; i++)
			{
				size_t bidx = 7 * i;
				float prob = boxes[bidx + 2];
				if (prob < MIN_PROB || prob > 1) continue;
				float clas = boxes[bidx + 1];
				float minx = boxes[bidx + 3] * WIDTH;
				float miny = boxes[bidx + 4] * WIDTH;
				float maxx = boxes[bidx + 5] * WIDTH;
				float maxy = boxes[bidx + 6] * WIDTH;
			

				const char* className = classNames[(uint32_t)clas];

				printf("%0.02f%% [[%0.01f, %0.01f],[%0.01f, %0.01f]] => %s\n", prob*100, minx, miny, maxx, maxy, className);
			}
			free(boxes);

			printf("inference time: %0.04f ms\n\n", ms);
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
