#include "pch.h"

int main(int argc, char* argv[])
{
	testing::InitGoogleTest(&argc, argv);
	testing::FLAGS_gtest_filter = "*.testing_*";

	RUN_ALL_TESTS();
}