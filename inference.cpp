#include "inference.h"

Model::Model(const char* filename)
{
	inputFrame.data = nullptr;
	outputFrame.data = nullptr;

	load(filename);
	setup();
}

Model::~Model()
{
	if (inputFrame.data) cudaFree(inputFrame.data);
	if (outputFrame.data) cudaFree(outputFrame.data);

	if (_context) _context->destroy();
	if (_engine)  _engine->destroy();
	if (_runtime) _runtime->destroy();
}

void Model::load(const char* filename)
{
	FILE* f = fopen(filename, "rb");
	if (!f) throw "Unable to open model";

	uint8_t* data = nullptr;
	size_t capacity = 0;
	size_t read=0;
	size_t incr = 1<<20;

	while (!feof(f))
	{
		if (capacity <= read)
		{
			capacity += incr;
			uint8_t* tmp = data;
			data = (uint8_t*) realloc(data, capacity);
			if (!data)
			{
				data = tmp;
				capacity -= incr;
				throw "Unable to reallocate model data";
			}
		}
		size_t r = fread(data + read, 1, capacity-read, f);
		if (r <= 0) throw "Error reading model data"; 
		read += r;
	}

	_runtime = createInferRuntime(*this);
	if (!_runtime) throw "Unable to create runtime";

	_engine = _runtime->deserializeCudaEngine(data, read, nullptr);
	if (!_engine) throw "Unable to deserialize engine";

	_context = _engine->createExecutionContext();
	if (!_context) throw "Unable to create execution context";
}

void Model::setup()
{
	int rc;
	printf("Setting up model input\n");
	idx.input = _engine->getBindingIndex("input");
	if (idx.input < 0) throw "Unable to find model input";
	if (DataType::kFLOAT != _engine->getBindingDataType(idx.input)) 
		throw "Unexpected datatype for input";
	dim.input = Dims4(1,3, 960, 540);
	_context->setBindingDimensions(idx.input, dim.input);

	printf("Setting up model output\n");
	idx.output = _engine->getBindingIndex("output");
	if (idx.output < 0) throw "Unable to find model output";
	if (DataType::kINT32 != _engine->getBindingDataType(idx.output))
		throw "Unexpected datatype for output";
	dim.output = _context->getBindingDimensions(idx.output);

	printf("Creating model frames\n");
	rc = cudaMalloc(&inputFrame.data, 3 * 960 * 540 * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate input frame device memory";

	rc = cudaMalloc(&outputFrame.data, 3 * 960 * 540 * sizeof(int));
	if (cudaSuccess != rc) throw "Unable to allocate input frame device memory";
}

void Model::infer(cudaStream_t stream)
{
	void* bindings[] = { inputFrame.data, outputFrame.data };
	bool result = _context->enqueueV2(bindings, stream, nullptr);
	if (!result) throw "Unable to enqueue inference";
}
