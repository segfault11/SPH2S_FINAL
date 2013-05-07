//------------------------------------------------------------------------------
//  Grid.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <cmath>
#include "cuda.h"
//------------------------------------------------------------------------------
struct Grid
{
    static unsigned int ComputeNumGridCells (const Grid& grid)
    {
        return grid.Dimensions.x * grid.Dimensions.y * grid.Dimensions.z;
    }

    static Grid MakeGrid(float3 origin, float3 end, float spacing)
    {
        // make a grid because non empty constructors are not allowed for CUDA

        Grid grid;
        grid.Origin = origin;
        grid.Spacing = spacing;

        // compute the dimensions of the grid
        grid.Dimensions.x = static_cast<int>(
                std::ceil((end.x - origin.x)/spacing)
            );
        grid.Dimensions.y = static_cast<int>(
                std::ceil((end.y - origin.y)/spacing)
            );
        grid.Dimensions.z = static_cast<int>(
                std::ceil((end.z - origin.z)/spacing)
            );
        return grid;
    }

    static Grid MakeGrid(float3 origin, int3 dimensions, float spacing)
    {
        // make a grid because non empty constructors are not allowed for CUDA

        Grid grid;
        grid.Origin = origin;
        grid.Dimensions = dimensions;
        grid.Spacing = spacing;
        return grid;
    }

    static float ComputeVolume(const Grid& grid)
    {
        // compute the volume of the grid

        return grid.Dimensions.x*grid.Dimensions.y*grid.Dimensions.z*
            grid.Spacing*grid.Spacing*grid.Spacing;
    }

    float3 Origin;
    int3 Dimensions;   // # of grid cells in each direction
    float Spacing;    
};
//------------------------------------------------------------------------------