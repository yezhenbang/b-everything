#pragma once
#include <fstream>

#ifdef UNICODE
typedef std::wfstream TFSTREAM;
#else
typedef std::fstream TFSTREAM;
#endif