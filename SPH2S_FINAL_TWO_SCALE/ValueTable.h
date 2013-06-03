//------------------------------------------------------------------------------
//  ValueTable.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <vector>
#include <string>
//------------------------------------------------------------------------------
class ValueTable
{
public:
    ValueTable();
    ~ValueTable();
    
    void AddValue(float x, float y);
    void Save(const std::string& filename);

private:
    std::vector<float> mXs;
    std::vector<float> mYs;
};
//------------------------------------------------------------------------------
