#pragma once

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
	Model(const char* filename, const char* prototxt, const char* caffemodel);
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
	} keepCount;

	virtual void log(Severity level, const char* msg) override
	{
		printf("[L%d] %s\n", (int) level, msg);
	}

	struct
	{
		Dims input;
		Dims boxes;
	} dim;

	struct
	{
		int input;
		int boxes;
	} idx;

	void infer(cudaStream_t);

	class Int8Calibrator : public IInt8EntropyCalibrator2
	{
	public:
		Int8Calibrator(size_t batchSize, size_t calibrationBatches);

		int getBatchSize() const override;
		bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
		const void* readCalibrationCache(size_t& length) override;
		void writeCalibrationCache(const void* data, size_t length) override;
	private:
		size_t _calibrationBatches;
		size_t _batchSize;
		size_t _batchAllocated;
		void* _devmem;
	};

protected:
	void load(const char* filename, const char* prototxt, const char* caffemodel);
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
