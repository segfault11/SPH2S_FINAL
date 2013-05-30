//------------------------------------------------------------------------------
//  Error.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <string>
//------------------------------------------------------------------------------
namespace CGTK {
namespace Error {
//------------------------------------------------------------------------------
enum ReportMode
{
    // Flags on how to report an error
    RM_EXCEPTION = 0,   // throw an runtime_exception
    RM_CONSOLE,         // dump message on console
    RM_LOG              // write message to log
};
//------------------------------------------------------------------------------
void SetReportMode(ReportMode mode);
void ReportError(const std::string& message);
void ReportWarning(const std::string& message);
void SetLogFileName(const std::string& filename);
void SetLogFileDirectory(const std::string& directory);
//------------------------------------------------------------------------------
} // end of namespace Error
} // end of namespace CGTK
//------------------------------------------------------------------------------
