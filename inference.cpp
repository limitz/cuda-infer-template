#include "inference.h"

Model::Model(const char* filename)
{
	inputFrame.data = nullptr;
	boxesFrame.data = nullptr;
	scoresFrame.data = nullptr;

	load(filename);
	setup();
}

Model::~Model()
{
	if (inputFrame.data) cudaFree(inputFrame.data);
	if (boxesFrame.data) cudaFree(boxesFrame.data);
	if (scoresFrame.data) cudaFree(scoresFrame.data);

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
	dim.input = Dims4(1,3, HEIGHT/SCALE, WIDTH/SCALE);
	_context->setBindingDimensions(idx.input, dim.input);

	printf("Setting up model output\n");
	idx.boxes = _engine->getBindingIndex("boxes");
	if (idx.boxes < 0) throw "Unable to find model output boxes";
	if (DataType::kFLOAT != _engine->getBindingDataType(idx.boxes))
		throw "Unexpected datatype for output boxes";
	dim.boxes = _context->getBindingDimensions(idx.boxes);

	idx.scores = _engine->getBindingIndex("classes");
	if (idx.scores < 0) throw "Unable to find model output boxes";
	if (DataType::kFLOAT != _engine->getBindingDataType(idx.scores))
		throw "Unexpected datatype for output scores";
	dim.scores = _context->getBindingDimensions(idx.scores);
	
	boxesFrame.length = dim.boxes.d[0] * dim.boxes.d[1] * dim.boxes.d[2];
	scoresFrame.length = dim.scores.d[0] * dim.scores.d[1] * dim.scores.d[2];
	printf("Creating model frames\n");
	rc = cudaMalloc(&inputFrame.data, 3 * WIDTH/SCALE * HEIGHT/SCALE * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate input frame device memory";

	rc = cudaMalloc(&boxesFrame.data, boxesFrame.length * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate boxes frame device memory";
	
	rc = cudaMalloc(&scoresFrame.data, scoresFrame.length * sizeof(float));
	if (cudaSuccess != rc) throw "Unable to allocate scores frame device memory";
}

void Model::infer(cudaStream_t stream)
{
	void* bindings[] = { inputFrame.data, boxesFrame.data, scoresFrame.data };
	bool result = _context->enqueueV2(bindings, stream, nullptr);
	if (!result) throw "Unable to enqueue inference";
}
