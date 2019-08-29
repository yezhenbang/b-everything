 /**
  * \brief catch unhandle crash and export a dump file
  * \usage SetUnhandledExceptionFilter(ApplicationCrashHandler);  in main()
  */
#pragma once

#ifndef B_CRASH_DUMP
#define B_CRASH_DUMP

#include <Windows.h>
#include <DbgHelp.h>
#include <string>
#include <tchar.h>

namespace b_util
{
	inline void CreateDumpFile(TCHAR* strDumpFilePathName, EXCEPTION_POINTERS* pException)
	{
		HANDLE hDumpFile = CreateFile(strDumpFilePathName, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL,
		                              NULL);
		MINIDUMP_EXCEPTION_INFORMATION dumpInfo;
		dumpInfo.ExceptionPointers = pException;
		dumpInfo.ThreadId = GetCurrentThreadId();
		dumpInfo.ClientPointers = TRUE;

		MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), hDumpFile, MiniDumpNormal, &dumpInfo, NULL, NULL);

		CloseHandle(hDumpFile);
	}

	inline LONG WINAPI ApplicationCrashHandler(LPEXCEPTION_POINTERS pException)
	{
		printf("UnHandleExceptionFun: Crash DMP.\n");
		TCHAR dump_name[128] = _T("./crash.dmp");

		CreateDumpFile(dump_name, pException);

		FatalAppExit(-1, dump_name);
		return EXCEPTION_EXECUTE_HANDLER;
	}
}
#endif
