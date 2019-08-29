#include "pch.h"
#include <tchar.h>
#include "../b_util/common/encrypt_tool.h"
#include "../b_util/string/string.h"


using namespace std;
using namespace b_util;

TEST(EncryptTest, testing_base64)
{
	TSTRING msg = _T("eyJIZWFkIjp7Ik1zZ1R5cGUiOiI4IiwiY29tcGxDb2QiOiIiLCJmaWxsMDMiOiIiLCJyZW1hcmsiOiIifSwiTWVzc2FnZSI6eyJDbE9yZElEIjoiMTAyMDAwMTc0NSIsIk9yZFN0YXR1cyI6IjAiLCJPcmRSZWpSZWFzb24iOiIiLCJFeGVjVHlwZSI6IjAifX0=");

	TCHAR buf[2048];
	base64_decode(msg.c_str(), buf, 2048);
	wcout << buf << endl;

	TCHAR enc_buf[2048];
	base64_encode(buf, enc_buf, 2048);
	wcout << enc_buf << endl;


	string str = "eyJIZWFkIjp7Ik1zZ1R5cGUiOiI4IiwiY29tcGxDb2QiOiIiLCJmaWxsMDMiOiIiLCJyZW1hcmsiOiIifSwiTWVzc2FnZSI6eyJDbE9yZElEIjoiMTAyMDAwMTc0NSIsIk9yZFN0YXR1cyI6IjAiLCJPcmRSZWpSZWFzb24iOiIiLCJFeGVjVHlwZSI6IjAifX0=";

	string ret = base64_decode(str);
	cout << ret << endl;
	str = base64_encode(ret);
	cout << str << endl;
}

