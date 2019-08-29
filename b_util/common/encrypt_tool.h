#pragma once

#ifndef B_ENCRYPT_TOOL
#define B_ENCRYPT_TOOL
#include <tchar.h>
#include <vector>
#include "../b_util.h"
#include "../string/string.h"

namespace b_util
{
	int encrypt_file(const TCHAR* in_path, const TCHAR* out_path, const TCHAR* encrypt_key);
	int decrypt_file(const TCHAR* in_path, const TCHAR* out_path, const TCHAR* encrypt_key);
	int read_encrypt_file(const TCHAR* in_path, TSTRING& out_buf, const TCHAR* encrypt_key);


	std::string base64_encode(const std::string& raw_msg);
	std::string base64_decode(const std::string& enc_msg);
	int base64_encode(const TCHAR* raw_msg, TCHAR* ret_buf, int ret_len);
	int base64_decode(const TCHAR* enc_msg, TCHAR* ret_buf, int ret_len);

	std::string trim_space(std::string src);

	std::string string_format(const char* pMsg, ...);

	int split_string(const std::string& str, const std::string& separator, std::vector<std::string>& str_vec);
}

#endif