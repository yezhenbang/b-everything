#include "pch.h"
#include "string.h"

std::string b_util::WString2String(const std::wstring& wstr)
{
	int nLen = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, NULL, NULL, NULL);
	char* chs = new char[nLen + 1];
	memset(chs, 0, nLen + 1);
	WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), nLen, chs, nLen, NULL, NULL);

	std::string str = chs;
	delete[] chs;
	return str;
}
