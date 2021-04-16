#include "inference.h"

Model::Model(const char* filename)
{
	inputFrame.data = nullptr;
	boxesFrame.data = nullptr;
	keepCount.data = nullptr;

	_context = nullptr;
	_engine = nullptr;
	_runtime = nullptr;

	load(filename);
	setup();
}

Model::~Model()
{
	if (inputFrame.data) cudaFree(inputFrame.data);
	if (boxesFrame.data) cudaFree(boxesFrame.data);
	if (keepCount.data) cudaFree(keepCount.data);

	if (_context) _context->destroy();
	if (_engine)  _engine->destroy();
	if (_runtime) _runtime->destroy();
}

void Model::load(const char* filename)
{
	/*
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
	*/
	initLibNvInferPlugins(this, "");

	_builder = createInferBuilder(*this);
	if (!_builder) throw "Unable to create builder";

	_network = _builder->createNetwork();
	if (!_network) throw "Unable to create network";

	_config = _builder->createBuilderConfig();
	if (!_config) throw "Unable to create config";

	_parser = createCaffeParser();
	if (!_parser) throw "Unable to create caffe parser";

	const IBlobNameToTensor* blobNameToTensor = _parser->parse(
			"../../models/ssd.prototxt", 
			"../../models/ssd.caffemodel",
			*_network,
			DataType::kFLOAT);

	_network->markOutput(*blobNameToTensor->find("keep_count"));
	_network->markOutput(*blobNameToTensor->find("detection_out"));
	_builder->setMaxBatchSize(1);
	_config->setMaxWorkspaceSize(36 * 1000 * 1000);
	//_config->setFlag(BuilderFlag::kFP16);
	
	_engine = _builder->buildEngineWithConfig(*_network, *_config);
	if (!_engine) throw "Unable to build engine";

	if (1 != _network->getNbInputs()) throw "Unexpected number of inputs";
	dim.input = _network->getInput(0)->getDimensions();
	if (3 != dim.input.nbDims) throw "Unexpected number of input dimensions";

	_context = _engine->createExecutionContext();
	if (!_context) throw "Unable to create execution context";
}

void Model::setup()
{
	int rc;
	
	inputFrame.pitch = 300 * sizeof(float);
	inputFrame.width = 300;
	inputFrame.height = 300;

	boxesFrame.length = 7 * 200;

	printf("Creating model frames\n");
	rc = cudaMalloc(&inputFrame.data, 3 * 300 * 300 * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate input frame device memory";

	rc = cudaMalloc(&boxesFrame.data, boxesFrame.length * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate boxes frame device memory";
	
	rc = cudaMalloc(&keepCount.data, sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate boxes frame device memory";
}

void Model::infer(cudaStream_t stream)
{
	void* bindings[] = { inputFrame.data, boxesFrame.data, keepCount.data };
	bool result = _context->enqueue(1, bindings, stream, nullptr);
	if (!result) throw "Unable to enqueue inference";

}
