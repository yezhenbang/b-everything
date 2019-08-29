#include "pch.h"

#include "boost_log.h"

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

using namespace b_util;

namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

//[ example_tutorial_file_advanced
void boost_log_init(const std::string& file_name)
{
	logging::add_file_log
	(
		keywords::file_name = file_name, /*< file name pattern >*/
		keywords::rotation_size = 10 * 1024 * 1024, /*< rotate files every 10 MiB... >*/
		keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/
		keywords::format = "[%TimeStamp%]: %Message%" /*< log record format >*/
	);

	logging::core::get()->set_filter
	(
		logging::trivial::severity >= logging::trivial::info
	);

	logging::add_common_attributes();
}

//]
//
//
//int main(int, char* [])
//{
//	using namespace logging::trivial;
//	src::severity_logger< severity_level > lg;
//
//	BOOST_LOG_SEV(lg, trace) << "A trace severity message";
//	BOOST_LOG_SEV(lg, debug) << "A debug severity message";
//	BOOST_LOG_SEV(lg, info) << "An informational severity message";
//	BOOST_LOG_SEV(lg, warning) << "A warning severity message";
//	BOOST_LOG_SEV(lg, error) << "An error severity message";
//	BOOST_LOG_SEV(lg, fatal) << "A fatal severity message";
//
//	return 0;
//}
