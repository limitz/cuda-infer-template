#include <asyncwork.h>

AsyncWorkQueue::AsyncWorkQueue(size_t threads, size_t capacity)
{
	_capacity = capacity;
	_nthreads = threads;

	_threads = (pthread_t*) calloc(sizeof(pthread_t), threads);
	if (!_threads) throw "Unable to allocate thread memory";

	_queue   = (AsyncWork**) calloc(sizeof(AsyncWork*), capacity);
	if (!_queue) throw "Unable to allocate queue memory";

	_head = _tail = 0;
	_active = true;
	pthread_mutex_init(&_lock, NULL);

	for (size_t i=0; i<threads; i++)
	{
		pthread_create(_threads + i, NULL, AsyncWorkQueue::consumerProc, this);
	}
}

int AsyncWorkQueue::enqueue(AsyncWork* work)
{
	pthread_mutex_lock(&_lock);

	if (_tail + 1 != _head) // do not overflow
	{
		_queue[_tail] = work;
		work->state = WORK_STATE_QUEUED;
		if (++_tail >= _capacity)
		{
			_tail = 0;
		}
		pthread_mutex_unlock(&_lock);
		return 0;
	}
	work->state = WORK_STATE_REJECTED;
	pthread_mutex_unlock(&_lock);
	return -ENOMEM;
}

AsyncWork* AsyncWorkQueue::getWork()
{
	pthread_mutex_lock(&_lock);
	AsyncWork* result = nullptr;
	if (_head != _tail)
	{
		result = _queue[_head];
		_queue[_head] = nullptr;
		if (++_head >= _capacity)
		{
			_head = 0;
		}
	}
	pthread_mutex_unlock(&_lock);
	return result;
}

AsyncWorkQueue::~AsyncWorkQueue()
{
	_active = false;
	for (size_t i=0; i<_nthreads; i++)
	{
		pthread_join(_threads[i], NULL);
	}
	free(_threads);
	
	pthread_mutex_destroy(&_lock);
	
	for (size_t i=0; i<_capacity; i++)
	{
		AsyncWork* work = _queue[i];
		if (work)
		{
			work->state = WORK_STATE_CANCELLED;
			switch (work->whenDone)
			{
				case WORK_DELETE: delete work; break;
				case WORK_FREE: free(work); break;
				default: break;
			}
		}
	}
	free(_queue);
}

void* AsyncWorkQueue::consumerProc(void* param)
{
	AsyncWorkQueue* self = (AsyncWorkQueue*) param;

	while (self->_active)
	{
		AsyncWork* work = self->getWork();
		if (!work)
		{
			usleep(1000);
			continue;
		}
		work->state = WORK_STATE_RUNNING;
		work->doWork();
		work->state = WORK_STATE_DONE;
		switch (work->whenDone)
		{
			case WORK_DELETE: delete work; break;
			case WORK_FREE: free(work); break;
			default: break;
		}
	}
	return nullptr;
}
