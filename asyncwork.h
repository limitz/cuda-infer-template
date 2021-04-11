#pragma once
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <stdio.h>

enum
{
	WORK_STATE_UNKNOWN,
	WORK_STATE_PENDING,
	WORK_STATE_QUEUED,
	WORK_STATE_REJECTED,
	WORK_STATE_RUNNING,
	WORK_STATE_DONE,
	WORK_STATE_CANCELLED,
};

enum
{
	WORK_KEEP,
	WORK_FREE,
	WORK_DELETE,
};

class AsyncWork
{
public:
	virtual void doWork() = 0;
	int state = WORK_STATE_PENDING;
	int whenDone = WORK_KEEP;

	virtual ~AsyncWork() {}

protected:
	AsyncWork()
	{
	}
};

class AsyncWorkQueue
{
public:
	AsyncWorkQueue(size_t threads, size_t capacity);
	~AsyncWorkQueue();
	int enqueue(AsyncWork* worker);
	void join();

protected:
	AsyncWork* getWork();

private:
	bool _active;
	size_t _head, _tail;
	size_t _capacity;
	size_t _nthreads;
	AsyncWork** _queue;
	pthread_mutex_t _lock;
	pthread_t* _threads;

	static void* consumerProc(void* param);
};
