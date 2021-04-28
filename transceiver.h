#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <pthread.h>

class UDPTransceiver
{
public:
	UDPTransceiver();
	~UDPTransceiver();

	void setRxCallback(
		void (*rxcb)
		(
			UDPTransceiver* trx, 
			const char* ip, 
			const uint8_t* data, 
			size_t size, 
			void* param 
		), void* param)
	{
		_rxcb = rxcb;
		_rxcb_param = param;
	}

	void setPort(uint16_t port)
	{
		_port = port;
	}

	void setMulticastAddress(const char* ip)
	{
		if (_multicastAddress) free(_multicastAddress);
		_multicastAddress = strdup(ip);
	}

	void start();
	void stop();
	void transmit(const char* data, size_t size)
	{
		transmit((const uint8_t*) data, size+1);
	}
	void transmit(const uint8_t* data, size_t size);
private:
	static void* process(void* instance);
	pthread_t _thread;
	struct sockaddr_in _listenerAddr;
	int _listener;
	struct sockaddr_in _senderAddr;
	int _sender;
	uint16_t _port;
	char* _multicastAddress;
	void* _rxcb_param;
	void (*_rxcb)(UDPTransceiver* trx, const char* ip, const uint8_t* data, size_t size, void* param);	
};

