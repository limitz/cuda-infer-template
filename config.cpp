#include <config.h>



void Config::loadFile(const char* filename)
{
	int rc;
	FILE* f = fopen(filename, "rb");
	if (!f) throw "Unable to open file";
	
	// these are arbitrary
	char* comment = nullptr;
	char* key = nullptr;
	char* val = nullptr;

	while (!feof(f))
	{
		char c = fgetc(f);
		if (c == '#')
		{
			if (!comment) comment = (char*) calloc(1<<16, 1);
			else strcat(comment, " ");
			if (!comment) throw "Unable to allocate memory for comment";
			rc = fscanf(f, " %[^\n]\n", comment + strlen(comment));
			if (rc <= 0) throw "Unable to parse comment in config";
		}
		else
		{
			ungetc(c, f);
			if (!key) key = (char*) calloc(64,1);
			if (!key) throw "Unable to allocate memory for key";
			if (!val) val = (char*) calloc(256,1);
			if (!val) throw "Unable to allocate memory for val";
			
			rc = fscanf(f, "%s %[^\n]\n", key, val);
			if (rc <= 0) throw "Unable to parse key value pair in config";

			int isPresentAt = -1;
			for (size_t i=0; i<_numEntries; i++)
			{
				if (!strcmp(_entries[i]._key, key))
				{
					isPresentAt = i;
					break;
				}
			}
			if (isPresentAt >= 0)
			{
				free(_entries[isPresentAt]._value);
				_entries[isPresentAt]._value = val;
				free(comment);
				free(key);
			}
			else
			{
				if (_numEntries >= CONFIG_MAX_ENTRIES) throw "Maximum number of entries reached";
				_entries[_numEntries]._key = key;
				_entries[_numEntries]._value = strdup(val);
				_entries[_numEntries]._defaultValue = val;
				_entries[_numEntries]._description = comment;
				_numEntries++;
			}


			comment = nullptr;
			key = nullptr;
			val = nullptr;
		}
	}
	fclose(f);


}

void Config::printHelp()
{
	printf("CONFIGURATION PARAMETERS\n");
	printf("------------------------\n");
	for (size_t i=0; i<_numEntries; i++)
	{
		printf("%-16s - %s. Default (%s)\n", _entries[i].key(), _entries[i].description(), _entries[i].defaultValue());
	}
	printf("\n");
}

void Config::print()
{
	printf("CONFIGURATION\n");
	printf("-------------\n");
	for (size_t i=0; i<_numEntries; i++)
	{
		printf("%-16s - %s\n", _entries[i].key(), _entries[i].string());
	}
	printf("\n");
}
