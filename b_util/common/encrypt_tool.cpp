#include "pch.h"
#include "encrypt_tool.h"
#include "../string/string.h"
#include "../file/file.h"


int b_util::encrypt_file(const TCHAR* in_path, const TCHAR* out_path, const TCHAR* encrypt_key)
{
	try
	{
		TFSTREAM fs(in_path, std::ios::in);
		TSTRING encrypt;
		int i = 0;
		int key_len = lstrlen(encrypt_key);
		TCHAR temp;
		while (fs.read(&temp, 1))
		{
			if (key_len == 0)
				encrypt += temp;
			else
				encrypt += TCHAR(temp ^ encrypt_key[i++ % key_len]);
		}
		fs.close();

		TFSTREAM out_fs(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (out_fs.is_open())
		{
			out_fs.write(encrypt.c_str(), encrypt.size());
			out_fs.close();
		}
	}
	catch (...)
	{
		return -1;
	}
	return 0;
}

int b_util::decrypt_file(const TCHAR* in_path, const TCHAR* out_path, const TCHAR* encrypt_key)
{
	try
	{
		TSTRING decrypt;
		read_encrypt_file(in_path, decrypt, encrypt_key);

		TFSTREAM out_fs(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (out_fs.is_open())
		{
			out_fs.write(decrypt.c_str(), decrypt.size());
			out_fs.close();
		}
	}
	catch (...)
	{
		return -1;
	}
	return 0;
}

int b_util::read_encrypt_file(const TCHAR* in_path, TSTRING& out_buf, const TCHAR* encrypt_key)
{
	try
	{
		TFSTREAM fs(in_path, std::ios::in | std::ios::binary);
		int key_len = lstrlen(encrypt_key);
		TSTRING decrypt;
		if (fs.is_open())
		{
			int i = 0;
			TCHAR temp;
			while (fs.read(&temp, 1))
			{
				if (key_len == 0)
					decrypt += temp;
				else
					decrypt += TCHAR(temp ^ encrypt_key[i++ % key_len]);
			}
		}
		fs.close();
		out_buf = std::move(decrypt);
	}
	catch (...)
	{
		out_buf = _T("");
		return -1;
	}
	return 0;
}

//base64
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>
#include <boost/archive/iterators/ostream_iterator.hpp>
#include <sstream>

const std::string base64_padding[] = {"", "==", "="};

std::string b_util::base64_encode(const std::string& raw_msg)
{
	using namespace boost::archive::iterators;

	std::stringstream os;
	// convert binary values to base64 characters
	typedef base64_from_binary
		// retrieve 6 bit integers from a sequence of 8 bit bytes
		<transform_width<const char*, 6, 8>> base64_enc; // compose all the above operations in to a new iterator

	std::copy(base64_enc(raw_msg.c_str()), base64_enc(raw_msg.c_str() + raw_msg.size()), ostream_iterator<char>(os));

	os << base64_padding[raw_msg.size() % 3];
	return os.str();
}

std::string b_util::base64_decode(const std::string& enc_msg)
{
	using namespace boost::archive::iterators;

	std::stringstream os;
	// convert binary values to base64 characters
	typedef transform_width
		// retrieve 6 bit integers from a sequence of 8 bit bytes
		<binary_from_base64<const char*>, 8, 6> base64_dec; // compose all the above operations in to a new iterator

	unsigned int size = enc_msg.size();
	if (size <= 0) return "";

	for (int i = 1; i <= 2; i++)
	{
		if (enc_msg.c_str()[size - 1] == '=')
			size--;
	}

	std::copy(base64_dec(enc_msg.c_str()), base64_dec(enc_msg.c_str() + size), ostream_iterator<char>(os));
	return os.str();
}

int b_util::base64_encode(const TCHAR* raw_msg, TCHAR* ret_buf, int ret_len)
{
	const TSTRING base64_chars = _T("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");

	int msg_len = lstrlen(raw_msg);
	TSTRING ret;
	int i = 0;
	TCHAR char_array_3[3];
	TCHAR char_array_4[4];

	while (msg_len--)
	{
		char_array_3[i++] = *(raw_msg++);
		if (i == 3)
		{
			char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
			char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
			char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
			char_array_4[3] = char_array_3[2] & 0x3f;

			for (i = 0; (i < 4); i++)
				ret += base64_chars[char_array_4[i]];
			i = 0;
		}
	}

	if (i)
	{
		for (int j = i; j < 3; j++)
			char_array_3[j] = '\0';

		char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
		char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
		char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
		char_array_4[3] = char_array_3[2] & 0x3f;

		for (int j = 0; (j < i + 1); j++)
			ret += base64_chars[char_array_4[j]];

		while ((i++ < 3))
			ret += '=';
	}

	lstrcpyn(ret_buf, ret.c_str(), ret_len);
	return 0;
}

inline bool is_base64(TCHAR c)
{
	return (isalnum(c) || (c == '+') || (c == '/'));
}

int b_util::base64_decode(const TCHAR* enc_msg, TCHAR* ret_buf, int ret_len)
{
	const TSTRING base64_chars = _T("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/");

	int msg_len = lstrlen(enc_msg);
	int i = 0;
	int in_ = 0;
	TCHAR char_array_4[4], char_array_3[3];
	TSTRING ret;

	while (msg_len-- && (enc_msg[in_] != '=') && is_base64(enc_msg[in_]))
	{
		char_array_4[i++] = enc_msg[in_];
		in_++;
		if (i == 4)
		{
			for (i = 0; i < 4; i++)
				char_array_4[i] = base64_chars.find(char_array_4[i]);

			char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
			char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
			char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

			for (i = 0; (i < 3); i++)
				ret += char_array_3[i];
			i = 0;
		}
	}

	if (i)
	{
		for (int j = i; j < 4; j++)
			char_array_4[j] = 0;

		for (int j = 0; j < 4; j++)
			char_array_4[j] = base64_chars.find(char_array_4[j]);

		char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
		char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
		char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

		for (int j = 0; (j < i - 1); j++) ret += char_array_3[j];
	}

	lstrcpyn(ret_buf, ret.c_str(), ret_len);
	return 0;
}


#include "boost/algorithm/string/trim_all.hpp"
std::string b_util::trim_space(std::string src)
{
	return boost::trim_right_copy(boost::trim_left_copy(src));
}

std::string b_util::string_format(const char* pMsg, ...)
{
	va_list args;
	char text[32768];// = {0};

	va_start(args, pMsg);
	vsnprintf(text, sizeof(text) - 1, pMsg, args);
	va_end(args);

	return std::string(text);
}

int b_util::split_string(const std::string& str, const std::string& separator, std::vector<std::string>& str_vec)
{
	std::string::size_type pos1 = 0;
	std::string::size_type pos2 = str.find(separator);

	while (std::string::npos != pos2)
	{
		str_vec.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + 1;
		pos2 = str.find(separator, pos1);
	}

	str_vec.push_back(str.substr(pos1));
	return str_vec.size();
}