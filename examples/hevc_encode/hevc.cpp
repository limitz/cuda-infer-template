#include <hevc.h>

int HEVCEncoder::getCapability(GUID id, NV_ENC_CAPS cap)
{
	if (!_encoder) throw "HEVC encoder not initialized";

	NV_ENC_CAPS_PARAM capsParam = { NV_ENC_CAPS_PARAM_VER };
	capsParam.capsToQuery = cap;
	
	int v;
	_nvenc.nvEncGetEncodeCaps(_encoder, id, &capsParam, &v);
	return v;
}

HEVCEncoder::HEVCEncoder(int deviceOrdinal, size_t width, size_t height, size_t fps)
{
	int rc;

	rc = cuDeviceGet(&_device, deviceOrdinal);
	if (CUDA_SUCCESS != rc) throw "Unable to get cuda device";

	rc = cuCtxCreate(&_context, 0, _device);
	if (CUDA_SUCCESS != rc) throw "Unable to create cuda context";

	_nvenc = { NV_ENCODE_API_FUNCTION_LIST_VER };

	rc = NvEncodeAPICreateInstance(&_nvenc);
	if (NV_ENC_SUCCESS != rc) throw "Unable to create NVENC instance";
	if (!_nvenc.nvEncOpenEncodeSessionEx) throw "NVENC API not found";

	
	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParams = { NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER };
	sessionParams.device = _context;
	sessionParams.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	sessionParams.apiVersion = NVENCAPI_VERSION;
	
	rc = _nvenc.nvEncOpenEncodeSessionEx(&sessionParams, &_encoder);
	if (NV_ENC_SUCCESS != rc) throw "Unable to open NVENC session";

	auto id = NV_ENC_CODEC_HEVC_GUID;
	printf("Encoder Capabilities\n");
	printf("- HEVC:          %d\n", getCapability(id, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES));
	printf("- HEVC_Main10:   %d\n", getCapability(id, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE));
	printf("- HEVC_Lossless: %d\n", getCapability(id, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE));
	printf("- HEVC_SAO:      %d\n", getCapability(id, NV_ENC_CAPS_SUPPORT_SAO));
	printf("- HEVC_444:      %d\n", getCapability(id, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE));
	printf("- HEVC_ME:       %d\n", getCapability(id, NV_ENC_CAPS_SUPPORT_MEONLY_MODE));
	printf("- HEVC_WxH:      %dx%d\n", 
			getCapability(id, NV_ENC_CAPS_WIDTH_MAX),
			getCapability(id, NV_ENC_CAPS_HEIGHT_MAX));
	printf("\n");

	// setup 
	auto preset = NV_ENC_PRESET_P3_GUID;
	NV_ENC_INITIALIZE_PARAMS initParams = { NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	initParams.encodeConfig = &encodeConfig;
	initParams.encodeGUID = id;
	initParams.presetGUID = preset;
	initParams.encodeWidth = width;
	initParams.encodeHeight = height;
	initParams.darWidth = width;
	initParams.darHeight = height;
	initParams.frameRateNum = fps;
	initParams.frameRateDen = 1;
	initParams.enablePTD = 1;
	initParams.reportSliceOffsets = 0;
	initParams.enableSubFrameWrite = 0;
	initParams.maxEncodeWidth = width;
	initParams.maxEncodeHeight = height;
	initParams.enableMEOnlyMode = false;
	initParams.enableOutputInVidmem = false;
	initParams.tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS; // NV_ENC_TUNING_INFO_HIGH_QUALITY

	NV_ENC_PRESET_CONFIG presetConfig = { NV_ENC_PRESET_CONFIG_VER, { NV_ENC_CONFIG_VER }};
	_nvenc.nvEncGetEncodePresetConfigEx(_encoder, id, preset, initParams.tuningInfo, &presetConfig);
	memcpy(initParams.encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));

	initParams.encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 0;//2
	initParams.encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
	initParams.encodeConfig->encodeCodecConfig.hevcConfig.idrPeriod = initParams.encodeConfig->gopLength;

	rc = _nvenc.nvEncInitializeEncoder(_encoder, &initParams);
	if (NV_ENC_SUCCESS != rc) throw "Unable to initialize HEVC encoder";
}

HEVCEncoder::~HEVCEncoder()
{
}
