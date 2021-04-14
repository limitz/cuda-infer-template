#pragma once

#ifndef WIDTH
#define WIDTH 1920
#endif

#ifndef HEIGHT
#define HEIGHT 1080
#endif

#ifndef SCALE
#define SCALE 4
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

class Model : public ILogger
{
public:
	Model(const char* filename);
	~Model();

	struct 
	{
		void* data;
		size_t pitch;
		size_t width;
		size_t height;
	} inputFrame;
	
	struct
	{
		void* data;
		size_t pitch;
		size_t width;
		size_t height;
	} outputFrame;

	virtual void log(Severity level, const char* msg) override
	{
		printf("[L%d] %s", (int) level, msg);
	}

	struct
	{
		Dims input;
		Dims output;
	} dim;

	struct
	{
		int input;
		int output;
	} idx;

	void infer(cudaStream_t);

protected:
	void load(const char* filename);
	void setup();

private:	
	ICudaEngine* _engine;
	IRuntime*    _runtime;
	IExecutionContext* _context;
};
