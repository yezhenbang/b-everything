#pragma once
#include <string>

#ifndef B_STRING
#define B_STRING

#ifdef UNICODE 
typedef std::wstring TSTRING;
#else
typedef std::string TSTRING;
#endif

namespace b_util
{
	std::string WString2String(const std::wstring& wstr);

}

#endif
