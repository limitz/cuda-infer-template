#include "inference.h"

Model::Model(const char* filename, const char* prototxt, const char* caffemodel)
{
	inputFrame.data = nullptr;
	boxesFrame.data = nullptr;
	keepCount.data = nullptr;

	_context = nullptr;
	_engine = nullptr;
	_runtime = nullptr;
	_network = nullptr;
	_config = nullptr;
	_builder = nullptr;
	_parser = nullptr;

	load(filename, prototxt, caffemodel);
	setup();
}

Model::~Model()
{
	if (inputFrame.data) cudaFree(inputFrame.data);
	if (boxesFrame.data) cudaFree(boxesFrame.data);
	if (keepCount.data) cudaFree(keepCount.data);

	if (_network) _network->destroy();
	if (_config)  _config->destroy();
	if (_builder) _builder->destroy();
	if (_parser) _parser->destroy();
	if (_context) _context->destroy();
	if (_engine)  _engine->destroy();
	if (_runtime) _runtime->destroy();
}

void Model::load(const char* filename, const char* prototxt, const char* caffemodel)
{
	int rc;
		
	initLibNvInferPlugins(this, "");
	
	FILE* f = fopen(filename, "rb");
	if (f)
	{
		printf("Found serialized engine %s\n", filename);

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
					fclose(f);
					throw "Unable to reallocate model data";
				}
			}
			rc = fread(data + read, 1, capacity-read, f);
			if (rc <= 0) 
			{
				fclose(f);
				throw "Error reading model data"; 
			}
			read += rc;
		}
		fclose(f);

		_runtime = createInferRuntime(*this);
		if (!_runtime) throw "Unable to create runtime";

		_engine = _runtime->deserializeCudaEngine(data, read, nullptr);
		if (!_engine) throw "Unable to deserialize engine";

		_context = _engine->createExecutionContext();
		if (!_context) throw "Unable to create execution context";
	}
	else
	{

		_builder = createInferBuilder(*this);
		if (!_builder) throw "Unable to create builder";

		_network = _builder->createNetworkV2(0);
		if (!_network) throw "Unable to create network";

		_config = _builder->createBuilderConfig();
		if (!_config) throw "Unable to create config";

		_parser = createCaffeParser();
		if (!_parser) throw "Unable to create caffe parser";

		const IBlobNameToTensor* blobNameToTensor = _parser->parse(
			prototxt, 
			caffemodel,
			*_network,
			DataType::kFLOAT);

		_network->markOutput(*blobNameToTensor->find("keep_count"));
		_network->markOutput(*blobNameToTensor->find("detection_out"));
		_builder->setMaxBatchSize(1);
		_config->setMaxWorkspaceSize(36 * 1000 * 1000);
		_config->setFlag(BuilderFlag::kFP16);
	
		if (false)
		{
			IInt8Calibrator* calibrator = new Int8Calibrator(1, 50);
			_config->setFlag(BuilderFlag::kINT8);
			_config->setInt8Calibrator(calibrator);
		}

		_engine = _builder->buildEngineWithConfig(*_network, *_config);
		if (!_engine) throw "Unable to build engine";

		f = fopen(filename, "wb");
		if (f)
		{
			IHostMemory* serialized = _engine->serialize();
			size_t written = 0;
			while (written < serialized->size())
			{
				rc = fwrite(((uint8_t*)serialized->data()) + written, 1, serialized->size() - written, f);
				if (ferror(f) || rc <= 0) throw "Unable to write serialized engine to file";
				written += rc;
			}
			printf("Successfully stored serialized engine\n");
			fclose(f);
			serialized->destroy();
		}
		else
		{
			fprintf(stderr, "Warning: unable to save serialized engine");
		}
		
		if (1 != _network->getNbInputs()) throw "Unexpected number of inputs";
		dim.input = _network->getInput(0)->getDimensions();
		if (3 != dim.input.nbDims) throw "Unexpected number of input dimensions";

		_context = _engine->createExecutionContext();
		if (!_context) throw "Unable to create execution context";
	}
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


Model::Int8Calibrator::Int8Calibrator(size_t batchSize, size_t calibrationBatches)
{
	_calibrationBatches = calibrationBatches;
	_batchSize = batchSize;
	_batchAllocated = _batchSize * 300 * 300 * 3 * sizeof(float);
	int rc = cudaMalloc(&_devmem, _batchAllocated);
	if (cudaSuccess != rc) throw "Unable to allocate calibration memory";
}


int Model::Int8Calibrator::getBatchSize() const
{ 
	return _batchSize; 
}


bool Model::Int8Calibrator::getBatch(void* bindings[], const char* /*names*/[], int /*nbBindings*/)
{
	if (0 == _calibrationBatches--) return false;
	
	void* tmp = malloc(_batchAllocated);
	cudaMemcpy(_devmem, tmp /* next batch */, _batchAllocated, cudaMemcpyHostToDevice);
	free(tmp);
	bindings[0] = _devmem;
	return true;
}

const void* Model::Int8Calibrator::readCalibrationCache(size_t& length)
{
	FILE* f = fopen("entropy8.cal", "rb");
	if (!f) return nullptr; //throw "Unable to open entropy input file";
	
	int rc = fread(&length, sizeof(size_t), 1, f);
	if (ferror(f) || rc != 1) throw "Unable to read header from entropy input file";

	void* result = malloc(length);
	if (!result) throw "Unable to allocate entropy memory";

	size_t read = 0;
	while (read < length)
	{
		rc = fread(((uint8_t*)result) + read, 1, length - read, f);
		if (ferror(f) || rc <= 0) throw "Unable to read entropy input file";
		read += rc;
	}
	fclose(f);
	return result;
}
		
void Model::Int8Calibrator::writeCalibrationCache(const void* data, size_t length)
{
	FILE* f = fopen("entropy8.cal","wb");
	if (!f) throw "Unable to open entropy output file";

	int rc = fwrite(&length, sizeof(size_t), 1, f);
	if (rc != 1) throw "Unable to write header to entropy output file";

	size_t written = 0;
	while (written < length)
	{
		rc = fwrite(((uint8_t*)data) + written, 1, length - written, f);
		if (ferror(f) || rc <= 0) throw "Unable to write entropy output file";
		written += rc;
	}
	fclose(f);
}
