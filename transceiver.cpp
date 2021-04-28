#include <transceiver.h>

UDPTransceiver::UDPTransceiver()
{
	_thread = 0;
	_listener = 0;
	_sender = 0;
	_port = 0;
	_rxcb = nullptr;
	_multicastAddress = nullptr;
}

UDPTransceiver::~UDPTransceiver()
{
	if (_multicastAddress) free(_multicastAddress);
	if (_thread) stop();
}

void UDPTransceiver::start()
{
	int rc;
	_listener = socket(AF_INET, SOCK_DGRAM, 0);
	if (_listener < 0) throw "Unable to create UDP listener socket";

	_sender = socket(AF_INET, SOCK_DGRAM, 0);
	if (_sender < 0) throw "Unable to create UDP sender socket";

	uint32_t yes = 1;
	rc = setsockopt(_listener, SOL_SOCKET, SO_REUSEADDR, (char*) &yes, sizeof(yes));
	if (rc < 0) throw "Unable to set socket option REUSEADDR";

	uint8_t loop = 0;
	rc = setsockopt(_sender, IPPROTO_IP, IP_MULTICAST_LOOP, &loop, sizeof(yes));
	if (rc < 0) throw "Unable to set socket option MULTICAST_LOOP";

	uint8_t ttl = 2;
	rc = setsockopt(_sender, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof(ttl));
	if (rc < 0) throw "Unable to set socket option MULTICAST_TTL";

	 
	memset(&_listenerAddr, 0, sizeof(_listenerAddr));
	_listenerAddr.sin_family = AF_INET;
	_listenerAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	_listenerAddr.sin_port = htons(_port);
	
	memset(&_senderAddr, 0, sizeof(_senderAddr));
	_senderAddr.sin_family = AF_INET;
	_senderAddr.sin_addr.s_addr = inet_addr(_multicastAddress);
	_senderAddr.sin_port = htons(_port);
	
	rc = bind(_listener, (struct sockaddr*) &_listenerAddr, sizeof(_listenerAddr));
	if (rc < 0) throw "Unable to bind UDP listener socket";

	struct ip_mreq mreq;
	mreq.imr_multiaddr.s_addr = inet_addr(_multicastAddress);
	mreq.imr_interface.s_addr = htonl(INADDR_ANY);
	rc = setsockopt(_listener, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&mreq, sizeof(mreq));
	if (rc < 0) throw "Unable to join UDP listener to multicast group";

	pthread_create(&_thread, NULL, UDPTransceiver::process, this);
}

void UDPTransceiver::stop()
{
	pthread_cancel(_thread);
	_thread = 0;
}

void UDPTransceiver::transmit(const uint8_t* data, size_t size)
{
	int rc = sendto(_sender, data, size, 0, (struct sockaddr*) &_senderAddr, sizeof(_senderAddr));
	if (rc < 0)
	{
		fprintf(stderr, "Error sending UDP message\n");
	}
}

void* UDPTransceiver::process(void* instance)
{
	UDPTransceiver* self = (UDPTransceiver*) instance;
	char ip[INET6_ADDRSTRLEN];

	while (1)
	{
		char msg[1<<14];
		struct sockaddr remoteAddr;
		socklen_t addrlen = sizeof(remoteAddr);
		int size = recvfrom(
				self->_listener, 
				msg, 1<<14, 0, 
				(struct sockaddr*) &remoteAddr, 
				&addrlen);
		if (size < 0)
		{
			fprintf(stderr, "Error receiving UDP message\n");
			break;
		}
		
		*ip = 0;
		switch (remoteAddr.sa_family)
		{
		case AF_INET:
			inet_ntop(AF_INET, &((struct sockaddr_in*)&remoteAddr)->sin_addr, ip, INET_ADDRSTRLEN);
			break;

		case AF_INET6:
			inet_ntop(AF_INET6, &((struct sockaddr_in6*)&remoteAddr)->sin6_addr, ip, INET6_ADDRSTRLEN);
			break;
		
		default:
			break;
		}
		if (self->_rxcb) self->_rxcb(self, ip, (uint8_t*)msg, size, self->_rxcb_param);
	}
	return nullptr;
}

