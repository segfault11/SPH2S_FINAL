//------------------------------------------------------------------------------
//  ParticleData.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include "OpenGL\OpenGL.h"
#include "Grid.h"
//------------------------------------------------------------------------------
class ParticleData
{
public:
    ParticleData (unsigned int maxParticles);
    ~ParticleData ();

    // set particle system for the use of cuda
    void Map ();

    // set particle system for the use of opengl
    void Unmap ();

    // creates a g box of particles
    static ParticleData* CreateParticleBox (const Grid& grid);
    static ParticleData* CreateParticleBoxCanvas (
        const Grid& grid, 
        int border
    );

    // the number of currently active particles
    unsigned int NumParticles;

    // the maximum amount of particles the system can store
    unsigned int MaxParticles;
    
    // CUDA part
    float* dPositions;
    float* dColorValues;

    // OpenGL part
    GLuint PositionsVBO;
    GLuint ColorValuesVBO;

private:
    mutable bool mIsMapped; // is the particle date mapped to CUDA memory?
    cudaGraphicsResource_t mGraphicsResources[2];
};
//------------------------------------------------------------------------------
