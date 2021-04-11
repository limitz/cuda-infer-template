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

#include <kinect.h>
#include <display.h>
#include <pthread.h>
#include <math.h>
#include <operators.h>
#include <asyncwork.h>
#include <jpegcodec.h>
#include <file.h>
#include <config.h>
#include <transceiver.h>

#ifndef TITLE
#define TITLE "CUDA INFERENCE DEMO"
#endif

#ifndef WIDTH
#define WIDTH 1920
#endif

#ifndef HEIGHT
#define HEIGHT 1080
#endif

#ifndef SCALE
#define SCALE 2
#endif

#ifndef USE_NVJPEG
#define USE_NVJPEG 0
#endif

#ifndef MIN
#define MIN(a,b) ((b) < (a) ? (b) : (a))
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
	if (x >= width || y >= height) return;

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
	if (x >= width || y >= height) return;
	size_t scstride = (width/SCALE) * (height/SCALE);
	size_t offset = y * width + x;
	size_t soffset = (y / SCALE) * (width/SCALE) + x / SCALE;

	normalized[soffset + 0 * scstride] = (rgb[offset*3 + 0]/255.0f - 0.485f) / (0.229f); 
	normalized[soffset + 1 * scstride] = (rgb[offset*3 + 1]/255.0f - 0.456f) / (0.224f); 
	normalized[soffset + 2 * scstride] = (rgb[offset*3 + 2]/255.0f - 0.406f) / (0.225f); 
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

class SaveKinectCapture : public AsyncWork
{
public:
	SaveKinectCapture(const char* filename, Kinect::Capture* capture)
	{
		_capture = capture;
		_filename = strdup(filename);
	}
	~SaveKinectCapture()
	{
		free(_filename);
		delete _capture;
	}
	virtual void doWork() override
	{
		char filenameWithExt[128];
		char header[128];
		sprintf(filenameWithExt, "%s.jpg", _filename);
		File colorFile(_capture->color.data, _capture->color.size, false);
		colorFile.save(filenameWithExt);

		sprintf(header, "P5\n# %s\n%lu %lu\n%d\n", 
				_filename, 
				_capture->transformedDepth.width, _capture->transformedDepth.height,
				0xFFFF);
		uint8_t* buffer = (uint8_t*)malloc(_capture->transformedDepth.size + strlen(header));
		memcpy(buffer, header, strlen(header));
		memcpy(buffer+strlen(header), _capture->transformedDepth.data, _capture->transformedDepth.size);

		sprintf(filenameWithExt, "%s.pgm.lz4", _filename);
		File depthFile(buffer, _capture->transformedDepth.size + strlen(header), true);
		depthFile.saveCompressed(filenameWithExt);
	}

private:
	Kinect::Capture* _capture;
	char* _filename;
};


//quick and dirty
static bool s_isStarted = false;
static const char* s_name = nullptr;
static bool s_isController = false;
static bool s_kinectShouldStart = false;
static bool s_kinectShouldStop = false;

static void onTransceiverRx(UDPTransceiver* trx, const char* ip, const uint8_t* message, size_t size, void* param)
{
	printf("[%s]: %s\n", ip, message);
	char* mstr = (char*) message;

	if (!s_isController)
	{
		if (!strcmp(mstr, "PING"))
		{
			trx->transmit(s_name, strlen(s_name));
		}
		if (!strcmp(mstr, "START"))
		{
			s_kinectShouldStart = true;
		}
		if (!strcmp(mstr, "STOP"))
		{
			s_kinectShouldStop = true;
		}
	}
}

static void* pingProc(void* param)
{
	while (1)
	{
		UDPTransceiver* tx = (UDPTransceiver*)param;
		tx->transmit("PING", 4);
		usleep(2000000);
	}
}

int main(int /*argc*/, char** /*argv*/)
{
	int rc;
	cudaStream_t stream = 0;

	try 
	{

		// LOAD CONFIG
		Config config;
		config.loadFile("global.config");
		try 
		{
			config.loadFile("local.config");
		}
		catch (const char* &e)
		{
			fprintf( stderr, "%s\n", e);
			printf( "WARN: local.config was not found or could not be read.\n"
				"WARN: using only settings specified in global.config.\n");
		}
		config.printHelp();
		config.print();

		// SETUP GPU
		printf("Setup GPU\n");
		selectGPU();
		
		rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
		if (cudaSuccess != rc) throw "Unable to create CUDA stream";

		// SETUP JPEG CODEC
		printf("Setup JPEG CODEC\n");
		JpegCodec codec;
		codec.prepare(WIDTH, HEIGHT, 3);
		rc = cudaMalloc(&imageBuffer, WIDTH * HEIGHT * 3);
		if (cudaSuccess != rc) throw "Unable to allocate image buffer";

		// SETUP DISPLAY
		printf("Setup display\n");
		CudaDisplay display(TITLE, WIDTH, HEIGHT); 
		cudaDeviceSynchronize();
		
		Kinect kinect;
		printf("Setup kinect\n");
		kinect.setFramesPerSecond(config.get("KinectFramesPerSecond").uint32());
		kinect.setColorResolution(config.get("KinectColorResolution").uint32());
		kinect.setDepthMode(
			config.get("KinectDepthModeNFOV").boolean(), 
			config.get("KinectDepthModeBinned").boolean());
		
		// SETUP ASYNCWORKQUEUE
		printf("Setup async work queue\n");
		AsyncWorkQueue work(4,1000);


		// SETUP TRANSCEIVER
		printf("Setup transceiver\n");
		s_name = config.get("NodeName").string();
		s_isController = config.get("IsController").boolean();
		UDPTransceiver transceiver;
		transceiver.setMulticastAddress(config.get("MulticastAddress").string());
		transceiver.setPort(config.get("MulticastPort").uint32());
		transceiver.setRxCallback(onTransceiverRx, &kinect);
		transceiver.start();	

		pthread_t pingThread = 0;
		if (s_isController)
		{
			pthread_create(&pingThread, nullptr, pingProc, &transceiver);
		}

		// READY TO RUN
		
		const char* outputDir = config.get("OutputDirectory").string();

		int frame_index = 0;
		char filename[128];
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
			if (s_kinectShouldStart && !s_isStarted)
			{
				s_isStarted = true;
				s_kinectShouldStart = false;
				kinect.start();
			}
			if (s_kinectShouldStop && s_isStarted)
			{
				s_isStarted = false;
				s_kinectShouldStop = false;
				kinect.stop();
			}

			auto capture = kinect.capture();

			if (capture)
			{
#if 0
#if USE_NVJPEG
				codec.decodeToDeviceMemoryGPU(
#else
				codec.decodeToDeviceMemoryCPU(
#endif
						imageBuffer, 
						capture->color.data, 
						capture->color.size, 
						stream);
#endif
				sprintf(filename, "%s/kinect_%04d", 
					outputDir,
					frame_index++);

				cudaStreamSynchronize(stream);
				auto savework = new SaveKinectCapture(filename,capture);
				work.enqueue(savework);
			}
#if 0
			f_jpeg<<<gridSize, blockSize, 0, stream>>>(
				display.CUDA.frame.data,
				display.CUDA.frame.pitch,
				imageBuffer,
				display.CUDA.frame.width,
				display.CUDA.frame.height
			);
#endif
			// copies the CUDA.frame.data to GL.pbaddr
			// and unmaps the GL.pbo
			display.cudaFinish(stream);
			display.render(stream);
			
			rc = cudaGetLastError();
			if (cudaSuccess != rc) throw "CUDA ERROR";

			// check escape pressed
			if (int e = display.events()) 
			{
				if (e < 0)
				{
					if (!config.get("IsController").boolean()) 
					{
						if (s_isStarted)	kinect.stop();
					}
					display.cudaUnmap(stream);
					cudaStreamDestroy(stream);
					return 0;
				}
				switch (e)
				{
					case 'b':
					case 'B':
						transceiver.transmit("START", 5);
						break;
					case 'e':
					case 'E':
						transceiver.transmit("STOP", 4);
						break;
				}
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
