#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifndef CONFIG_MAX_ENTRIES
#define CONFIG_MAX_ENTRIES 256
#endif

class Config
{
public:
	struct ConfigEntry
	{
	public:
		const char* key() const { return _key; }
		const char* description() const { return _description; }
		const char* defaultValue() const { return _defaultValue; }

		bool boolean() const { return strlen(_value) && (*_value != '0') && strcmp(_value, "no") && strcmp(_value, "false"); }
		const char* string() const { return _value; }
		int32_t int32() const   { return (int32_t)  atoi(_value); }
		uint32_t uint32() const { return (uint32_t) atoi(_value); }
	private:
		friend class Config;
		char* _key;
		char* _value;
		char* _description;
		char* _defaultValue;

	};

	Config()
	{
		_entries = (ConfigEntry*) calloc(sizeof(ConfigEntry), CONFIG_MAX_ENTRIES);
		_numEntries = 0;
	}
	
	~Config()
	{
		free(_entries);
	}

	void loadFile(const char* filename);
	void print();
	void printHelp();

	const ConfigEntry& get(const char* key) 
	{
		for (size_t i=0; i<_numEntries; i++)
		{
			if (!strcmp(_entries[i].key(), key)) return _entries[i];
		}
		fprintf(stderr, "Config did not contain an entry for key \"%s\"", key);
		throw "Config key not found";
	}
	const ConfigEntry* entries() const { return _entries; }
	size_t count() const { return _numEntries; }
private:
	ConfigEntry* _entries;
	size_t _numEntries;
};
