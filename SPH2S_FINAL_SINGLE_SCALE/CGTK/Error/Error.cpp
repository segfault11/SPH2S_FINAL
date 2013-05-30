//------------------------------------------------------------------------------
//  Error.cpp
//------------------------------------------------------------------------------
#include <iostream>
#include "Error.h"
//------------------------------------------------------------------------------
void messageErrorHandler(const std::string& message);
//------------------------------------------------------------------------------
static void (*errorHandler)(const std::string& message) = messageErrorHandler;
//------------------------------------------------------------------------------
void CGTK::Error::ReportError(const std::string& message)
{
    (*errorHandler)(message);
}
//------------------------------------------------------------------------------
void messageErrorHandler(const std::string& message)
{
    std::cerr << message << std::endl;
}
//------------------------------------------------------------------------------
