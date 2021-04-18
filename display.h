#pragma once

#ifndef JETSON
#define JETSON 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

#include <EGL/egl.h>

#if JETSON
#include <GLES3/gl32.h>
#else
#include <GLES2/gl2.h>
#include <EGL/eglext.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class CudaDisplay
{	
public:
	CudaDisplay(const char* title, size_t width, size_t height);
	~CudaDisplay();

	struct 
	{
		int handle;
		int width;
		int height;
		Window window;
		Window root;
		Display* display;
	} X;

	struct 
	{
		EGLDisplay display = EGL_NO_DISPLAY;
		EGLSurface surface = EGL_NO_SURFACE;
		EGLContext context = EGL_NO_CONTEXT;
	} EGL;

	struct 
	{
		GLuint vao = 0;
		GLuint pbo = 0;
		GLuint vbo = 0;
		
		GLuint texture = 0;
		GLuint program = 0;
		cudaGraphicsResource* pbres = NULL;
		float4* pbaddr = NULL;
		size_t pbsize = 0;
	} GL;

	struct
	{
		struct 
		{
			float4* data;
			size_t width;
			size_t height;
			size_t pitch;
		} frame;
	} CUDA;

	int attachShader(GLenum type, const char* path);
	int makeProgram();
	int initBuffers();
	int swapBuffers();
	int cudaMap(cudaStream_t stream);
	int cudaFinish(cudaStream_t stream);
	int cudaUnmap(cudaStream_t stream);
	int close();
	int render(cudaStream_t stream);
	int events();
};
