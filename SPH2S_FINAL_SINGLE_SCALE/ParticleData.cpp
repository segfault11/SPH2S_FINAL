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
 }
//------------------------------------------------------------------------------
ParticleData::~ParticleData ()
{
    glDeleteBuffers(1, &PositionsVBO);
    glDeleteBuffers(1, &ColorValuesVBO);
}
//------------------------------------------------------------------------------
void ParticleData::Map ()
{
    CUDA::GL::MapResources(
        2, 
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
}
//------------------------------------------------------------------------------
void ParticleData::Unmap ()
{
    CUDA::GL::UnmapResources(2, mGraphicsResources);
}
//------------------------------------------------------------------------------
ParticleData* ParticleData::Union(
    const ParticleData* a, 
    const ParticleData* b
)
{
    // Creates a new ParticleData object that contains particle data of two 
    // other ParticleData objects.

    if (a == NULL || b == NULL)
    {
        return NULL;
    }

    unsigned int maxParticles = a->MaxParticles + b->MaxParticles;
    
    // allocate new particle data
    ParticleData* result = new ParticleData(maxParticles);

    // copy positions data from particle data a and b to the result buffer
    // object
    glBindBuffer(GL_COPY_WRITE_BUFFER, result->PositionsVBO);
    glBindBuffer(GL_COPY_READ_BUFFER, a->PositionsVBO);
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, 
        GL_COPY_WRITE_BUFFER,
        0,
        0,
        sizeof(float)*3*a->NumParticles
    );
    glBindBuffer(GL_COPY_READ_BUFFER, b->PositionsVBO);
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, 
        GL_COPY_WRITE_BUFFER,
        0,
        sizeof(float)*3*a->NumParticles,
        sizeof(float)*3*b->NumParticles
    );

    // copy color values data from particle data a and b to the result buffer
    // object
    glBindBuffer(GL_COPY_WRITE_BUFFER, result->ColorValuesVBO);
    glBindBuffer(GL_COPY_READ_BUFFER, a->ColorValuesVBO);
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, 
        GL_COPY_WRITE_BUFFER,
        0,
        0,
        sizeof(float)*a->NumParticles
    );
    glBindBuffer(GL_COPY_READ_BUFFER, b->ColorValuesVBO);
    glCopyBufferSubData(
        GL_COPY_READ_BUFFER, 
        GL_COPY_WRITE_BUFFER,
        0,
        sizeof(float)*a->NumParticles,
        sizeof(float)*b->NumParticles
    );

    return result;
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
    glBufferData(
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

