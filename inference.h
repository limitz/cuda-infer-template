#pragma once

#ifndef WIDTH
#define WIDTH 1920
#endif

#ifndef HEIGHT
#define HEIGHT 1080
#endif

#ifndef SCALE
#define SCALE 2
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;

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
		size_t length;
	} boxesFrame;

	struct 
	{
		void* data;
		size_t length;
	} scoresFrame;

	virtual void log(Severity level, const char* msg) override
	{
		printf("[L%d] %s\n", (int) level, msg);
	}

	struct
	{
		Dims input;
		Dims boxes;
		Dims scores;
	} dim;

	struct
	{
		int input;
		int boxes;
		int scores;
	} idx;

	void infer(cudaStream_t);

protected:
	void load(const char* filename);
	void setup();

private:	
	ICudaEngine* _engine;
	IRuntime*    _runtime;
	IExecutionContext* _context;
	INetworkDefinition* _network;
	IBuilder* _builder;
	IBuilderConfig* _config;
	ICaffeParser* _parser;
};
