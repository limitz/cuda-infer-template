#include <display.h>

#ifndef FULLSCREEN
#define FULLSCREEN 1
#endif

CudaDisplay::CudaDisplay(const char* title, size_t width, size_t height)
{
	int rc;

	CUDA.frame.data = nullptr;
	CUDA.frame.width = width;
	CUDA.frame.height = height;
	rc = cudaMallocPitch(&CUDA.frame.data, &CUDA.frame.pitch, width * sizeof(*CUDA.frame.data), height);
	if (cudaSuccess != rc) throw "Unable to allocate device memory for display";

	X.width = width;
	X.height = height;

	XSetWindowAttributes window_attr;
	memset(&window_attr, 0, sizeof(window_attr));

	EGLConfig config = {0};
	EGLint num_config;
	EGLint config_attr[] = {
		EGL_RED_SIZE,   1,
		EGL_GREEN_SIZE, 1,
		EGL_BLUE_SIZE,  1,
#ifndef JETSON
		EGL_DEPTH_SIZE, 1,
#else
		EGL_DEPTH_SIZE, 16,
		EGL_SAMPLE_BUFFERS, 0,
		EGL_SAMPLES, 0,
#endif
		EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
		EGL_NONE
	};
	EGLint context_attr[] = {
		EGL_CONTEXT_CLIENT_VERSION, 3,
		EGL_NONE
	};


	if (!(X.display = XOpenDisplay(NULL))) 
		throw "XOpenDisplay";
	
	X.handle = DefaultScreen(X.display);
	
	if (!(EGL.display = eglGetDisplay(X.display))) 
		throw "eglGetDisplay";
	
	if (!eglInitialize(EGL.display, 0, 0)) 
		throw "eglInitialize";
	
	if (!eglChooseConfig(EGL.display, config_attr, &config, 1, &num_config))
		throw "eglChooseConfig";

	if (!num_config)
		throw "num_config";
	
	X.root = RootWindow(X.display, X.handle);
#ifndef JETSON
	XVisualInfo *visualInfo, visualTemplate;
	
	if (!eglGetConfigAttrib(EGL.display, config, EGL_NATIVE_VISUAL_ID, (EGLint*)&visualTemplate.visualid))
		throw "eglGetConfigAttrib";

#ifdef THROW_EXCEPTION_ON_XGETVISUALINFO_FAILED
	// This often fails for some reason, but program works even if it does.
	int num_visuals;
	if(!(visualInfo = XGetVisualInfo(X.display, VisualIDMask, &visualTemplate, &num_visuals)))	
		throw "XGetVisualInfo";
#else
	visualInfo = XGetVisualInfo(X.display, VisualIDMask, &visualTemplate, &num_visuals);
#endif

	window_attr.background_pixel = 0;
	window_attr.border_pixel = 0;
	window_attr.colormap = XCreateColormap(X.display, X.root, visualInfo->visual, AllocNone);
	window_attr.event_mask = StructureNotifyMask | ExposureMask | KeyPressMask;
	X.window = XCreateWindow(X.display, X.root, 0, 0, X.width, X.height, 0,
			visualInfo->depth, InputOutput, visualInfo->visual,
			CWBackPixel | CWBorderPixel | CWColormap | CWEventMask,
			&window_attr);
	XFree(visualInfo);
#else
	X.window = XCreateSimpleWindow(X.display, X.root, 
			0, 0, X.width, X.height, 0,
			BlackPixel(X.display, X.handle),
			BlackPixel(X.display, X.handle));

	XSelectInput(X.display, X.window, ExposureMask | StructureNotifyMask | KeyPressMask);
#endif

	XStoreName(X.display, X.window, title);
	XMapWindow(X.display, X.window);
	XFlush(X.display);

#if FULLSCREEN
	XEvent	x11_event;
	Atom	x11_state_atom;
	Atom	x11_fs_atom;

	x11_state_atom	= XInternAtom( X.display, "_NET_WM_STATE", False );
	x11_fs_atom	= XInternAtom( X.display, "_NET_WM_STATE_FULLSCREEN", False );

	x11_event.xclient.type		= ClientMessage;
	x11_event.xclient.serial	= 0;
	x11_event.xclient.send_event	= True;
	x11_event.xclient.window	= X.window;
	x11_event.xclient.message_type	= x11_state_atom;
	x11_event.xclient.format	= 32;
	x11_event.xclient.data.l[ 0 ]	= 1;
	x11_event.xclient.data.l[ 1 ]	= x11_fs_atom;
	x11_event.xclient.data.l[ 2 ]	= 0;

	XSendEvent(X.display, X.root, False, SubstructureRedirectMask | SubstructureNotifyMask, &x11_event);
#endif
	
	eglBindAPI(EGL_OPENGL_ES_API);
	
	if (!(EGL.context = eglCreateContext(EGL.display, config, 0, context_attr)))
		throw "eglCreateContext";

	if (!(EGL.surface = eglCreateWindowSurface(EGL.display, config, (EGLNativeWindowType) X.window, NULL)))
		throw "eglCreateWindowSurface";


	if (!eglMakeCurrent(EGL.display,EGL.surface, EGL.surface, EGL.context))
		throw "eglMakeCurrent";

	initBuffers();
	makeProgram();
	
	eglSwapInterval(EGL.display, 1);
}

int CudaDisplay::attachShader( GLenum type, const char* path)
{
	auto f = fopen(path, "rb");
	if (!f) throw "Unable to open shader";

	fseek(f, 1, SEEK_END);
	GLint size = ftell(f);
	char *src = (char*)malloc(size + 1);
	memset(src, 0, size + 1);
	fseek(f, 0, SEEK_SET);	
	fread(src, 1, size, f);
	
	fclose(f);

	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, (const GLchar**)&src, &size);
	glCompileShader(shader);

	GLint compiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

	if (GL_FALSE == compiled)
	{
		fprintf(stderr, "%s -> Compilation failed.",path);
		GLint maxLength = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
		
		char errorLog[maxLength];
		glGetShaderInfoLog(shader, maxLength, &maxLength, errorLog);
		fprintf(stderr, "%s", errorLog);
		throw "Compilation failed";
	}

	glAttachShader(GL.program, shader);
	glDeleteShader(shader);
	free(src);
	return 0;
}

int CudaDisplay::makeProgram()
{
	GL.program = glCreateProgram();
	attachShader(GL_VERTEX_SHADER, "vertex.glsl");
	attachShader(GL_FRAGMENT_SHADER, "fragment.glsl");
	glLinkProgram(GL.program);

	GLint linked;
	glGetProgramiv(GL.program, GL_LINK_STATUS, &linked);
	if (GL_FALSE == linked)
	{
		fprintf(stderr, "Linking failed\n");
		GLint maxLength = 0;
		glGetShaderiv(GL.program,GL_INFO_LOG_LENGTH, &maxLength);
		char errorLog[maxLength];
		glGetShaderInfoLog(GL.program, maxLength, &maxLength, errorLog);
		fprintf(stderr,"%s", errorLog);
		throw "Linking failed";
	}
		
	return 0;
}

int CudaDisplay::initBuffers()
{
	GLfloat quad[] = {
		-1, -1, 
		 1, -1,
		-1,  1,
		 1,  1
	};

	glEnableVertexAttribArray(0);
	glGenBuffers(1, &GL.vbo);
	glBindBuffer(GL_ARRAY_BUFFER, GL.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)0, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glGenBuffers(1, &GL.pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, GL.pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 
		X.width * X.height * sizeof(float4),
		0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&GL.pbres, GL.pbo, 
		cudaGraphicsMapFlagsWriteDiscard);

	glGenTextures(1, &GL.texture);
	glBindTexture(GL_TEXTURE_2D, GL.texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, X.width, X.height, 0, 
			GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
	return 0;
}

int CudaDisplay::cudaMap(cudaStream_t /*stream*/)
{
	int rc;
	rc = cudaGraphicsMapResources(1, &GL.pbres, 0);
	if (cudaSuccess != rc) throw "Unable to map resource";

	rc = cudaGraphicsResourceGetMappedPointer((void**)&GL.pbaddr, &GL.pbsize, GL.pbres);
	if (cudaSuccess != rc) throw "Unable to get mapped pointer";
	
	
	return 0;
}

int CudaDisplay::cudaFinish(cudaStream_t stream)
{
	int rc;
	rc = cudaMemcpy2DAsync(
		GL.pbaddr, CUDA.frame.width * sizeof(*CUDA.frame.data),
		CUDA.frame.data, CUDA.frame.pitch,
		CUDA.frame.width * sizeof(*CUDA.frame.data),
		CUDA.frame.height,
		cudaMemcpyDeviceToHost,
		stream);
	if (cudaSuccess != rc) throw "Unable to memcpy device memory to mapped pointer";
	return 0;
}

int CudaDisplay::cudaUnmap(cudaStream_t stream)
{
	int rc;
	rc = cudaStreamSynchronize(stream);
	if (cudaSuccess != rc) throw "Unable to synchronize stream";

	rc = cudaGraphicsUnmapResources(1, &GL.pbres, 0);
	if (cudaSuccess != rc) throw "Unable to unmap resources";
	return 0;
}

int CudaDisplay::render(cudaStream_t stream)
{
	cudaStreamSynchronize(stream);
	glClearColor(0.5, 0.5, 0.5, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glActiveTexture(0);
	glBindTexture(GL_TEXTURE_2D, GL.texture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, GL.pbo);
	glTexSubImage2D(GL_TEXTURE_2D, 0,
			0,0,X.width, X.height, 
			GL_RGBA, GL_FLOAT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

	glUseProgram(GL.program);
	glEnableVertexAttribArray(0);
	glUniform1i(glGetUniformLocation(GL.program, "texdata"), 0);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glFinish();

	swapBuffers();
	return 0;
}

int CudaDisplay::events()
{
	int rc = 0;
	XEvent event;
	KeySym key;
	char text[256];
	while (XPending(X.display))
	{
		XNextEvent(X.display, &event);
		switch (event.type)
		{
		case Expose:
			if (0 == event.xexpose.count) 
			{ 
				printf("redraw\n"); 
			}
			break;

		case KeyPress:
			if (1 == XLookupString(&event.xkey, text, 256, &key, 0))
			{
				if (27 == *text)
				{
					printf("Escape pressed\n");
					return 1;
				}
				printf("Keypress: %c (%d)\n", *text, *text);
			}
			break;

		case ButtonPress:
			printf("Mouse press %d at (%d, %d)\n",
				event.xbutton.button,
				event.xbutton.x,
				event.xbutton.y);
			break;
		
		case ButtonRelease:
			printf("Mouse release %d at (%d, %d)\n",
				event.xbutton.button,
				event.xbutton.x,
				event.xbutton.y);
			break;

		case MotionNotify:
			//something like:
			//x = event.xmotion.x;
			//y = event.xmotion.y;
			//if (screen->capture)
			//{
			//	screen->cur_dx = x - screen->cur_x;
			//	screen->cur_dy = y - screen->cur_y;
			//	screen->set_dx = x - screen->set_x;
			//	screen->set_dy = y - screen->set_y;
			//}
			//screen->cur_x = x;
			//screen->cur_y = y;

			break;
		}
	}
	return rc;
}

int CudaDisplay::swapBuffers()
{
	eglSwapBuffers(EGL.display, EGL.surface);
	return 0;
}

CudaDisplay::~CudaDisplay()
{
	if (CUDA.frame.data)
	{
		cudaFree(CUDA.frame.data);
	}
	if (EGL.display != EGL_NO_DISPLAY)
	{
		eglMakeCurrent(EGL.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (EGL.context != EGL_NO_CONTEXT)
		{
			eglDestroyContext(EGL.display, EGL.context);
		}
		if (EGL.surface != EGL_NO_SURFACE)
		{
			eglDestroySurface(EGL.display, EGL.surface);
		}
	}
	if (X.window)
	{
		XDestroyWindow(X.display, X.window);
	}
	XCloseDisplay(X.display);
}
