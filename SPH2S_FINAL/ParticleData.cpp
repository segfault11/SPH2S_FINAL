//------------------------------------------------------------------------------
//  ParticleData.h
//------------------------------------------------------------------------------
#include "ParticleData.h"
#include "cuda.h"
#include <iostream>
//------------------------------------------------------------------------------
ParticleData::ParticleData (unsigned int maxParticles)
: MaxParticles(maxParticles), NumParticles(maxParticles), mIsMapped(false)
{
    // initilizes the particle data, this constructor sets all particles active

    // allocate vertex buffer objects for the positions, colors and the ids of 
    // the particles, after allocation register them with cuda
    GL::CreateBufferObject(
        PositionsVBO, 
        GL_ARRAY_BUFFER, 
        sizeof(float)*3*maxParticles, 
        NULL, 
        GL_DYNAMIC_DRAW
    );
    CUDA::GL::RegisterBuffer(
        &mGraphicsResources[0], 
        PositionsVBO, 
        cudaGraphicsMapFlagsNone
    );
    GL::CreateBufferObject(
        ColorValuesVBO, 
        GL_ARRAY_BUFFER, 
        sizeof(float)*maxParticles, 
        NULL, 
        GL_DYNAMIC_DRAW
    );
    CUDA::GL::RegisterBuffer(
        &mGraphicsResources[1], 
        ColorValuesVBO, 
        cudaGraphicsMapFlagsNone
    );
 
    // in the beginning all particles are active
    unsigned int* activeIDs = new unsigned int[maxParticles];
    for (unsigned int i = 0; i < maxParticles; i++)
    {
        activeIDs[i] = i;
    }
    GL::CreateBufferObject(
        ActiveIdsVBO, 
        GL_ELEMENT_ARRAY_BUFFER, 
        sizeof(unsigned int)*maxParticles, 
        activeIDs, 
        GL_DYNAMIC_DRAW
    );
    CUDA::GL::RegisterBuffer(
        &mGraphicsResources[2], 
        ActiveIdsVBO, 
        cudaGraphicsMapFlagsNone
    );
    delete[] activeIDs;
}
//------------------------------------------------------------------------------
ParticleData::~ParticleData ()
{
    glDeleteBuffers(1, &PositionsVBO);
    glDeleteBuffers(1, &ColorValuesVBO);
    glDeleteBuffers(1, &ActiveIdsVBO);
}
//------------------------------------------------------------------------------
void ParticleData::Map ()
{
    CUDA::GL::MapResources(
        3, 
        mGraphicsResources
    );
    CUDA::GL::GetMappedPointer<float>(
        &dPositions, 
        mGraphicsResources[0]
    );
    CUDA::GL::GetMappedPointer<float>(
        &dColorValues, 
        mGraphicsResources[1]
    );
    CUDA::GL::GetMappedPointer<unsigned int>(
        &dActiveIDs, 
        mGraphicsResources[2]
    );
}
//------------------------------------------------------------------------------
void ParticleData::Unmap ()
{
    CUDA::GL::UnmapResources(3, mGraphicsResources);
}
//------------------------------------------------------------------------------
ParticleData* ParticleData::CreateParticleBox (const Grid& grid)
{
    unsigned int numParticles = Grid::ComputeNumGridCells(grid);
    float* pos = new float[3*numParticles];

    // initialize positions
    unsigned int c = 0;
    for (int k = 0; k < grid.Dimensions.z; k++)
    {
        for (int j = 0; j < grid.Dimensions.y; j++)
        {
            for (int i = 0; i < grid.Dimensions.x; i++)
            {
                pos[c + 0] = grid.Origin.x + i*grid.Spacing;
                pos[c + 1] = grid.Origin.y + j*grid.Spacing;
                pos[c + 2] = grid.Origin.z + k*grid.Spacing;

                c += 3;
            }    
        }
    }

    // create particle data and the vertex buffer object for the positions
    ParticleData* data = new ParticleData(numParticles);
    glBindBuffer(GL_ARRAY_BUFFER, data->PositionsVBO);
    glBufferData
    (
        GL_ARRAY_BUFFER, 
        sizeof(float)*3*numParticles, 
        pos, 
        GL_DYNAMIC_DRAW
    );
    delete[] pos;
    return data;    
}
//------------------------------------------------------------------------------
ParticleData* ParticleData::CreateParticleBoxCanvas (
    const Grid& grid, 
    int border
)
{
    unsigned int numParticles = (grid.Dimensions.x + 2*border)*
        (grid.Dimensions.y + 2*border)*(grid.Dimensions.z + 2*border);
    numParticles -= Grid::ComputeNumGridCells(grid);
    float* pos = new float[3*numParticles];

    unsigned int c = 0;
    for (int k = -border; k < grid.Dimensions.z + border; k++)
    {
        for (int j = -border; j < grid.Dimensions.y + border; j++)
        {
            for (int i = -border; i < grid.Dimensions.x + border; i++)
            {

                if (!((i >= 0 && i < grid.Dimensions.x) && 
                    (j >= 0 && j < grid.Dimensions.y) && 
                    (k >= 0 && k < grid.Dimensions.z)))
                {
                    pos[c + 0] = grid.Origin.x + i*grid.Spacing;
                    pos[c + 1] = grid.Origin.y + j*grid.Spacing;
                    pos[c + 2] = grid.Origin.z + k*grid.Spacing;
                    c += 3;
                }
            }    
        }
    }

    // create particle data and the vertex buffer object for the positions
    ParticleData* data = new ParticleData(numParticles);
    glBindBuffer(GL_ARRAY_BUFFER, data->PositionsVBO);
    glBufferData
    (
        GL_ARRAY_BUFFER, 
        sizeof(float)*3*numParticles, 
        pos, 
        GL_DYNAMIC_DRAW
    );
    delete[] pos;
    return data; 

}
//------------------------------------------------------------------------------

