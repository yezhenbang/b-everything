#pragma once
#include <string>

namespace b_util
{
#define LOG_E	BOOST_LOG_TRIVIAL(error)
#define LOG_I	BOOST_LOG_TRIVIAL(info)
#define LOG_D	BOOST_LOG_TRIVIAL(debug)
#define LOG_T	BOOST_LOG_TRIVIAL(trace)

	//[ example_tutorial_file_advanced
	void boost_log_init(const std::string& file_name);
}
