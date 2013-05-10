//------------------------------------------------------------------------------
//  Solver.cu
//------------------------------------------------------------------------------
#include "Solver.h"
//------------------------------------------------------------------------------

//==============================================================================
//  CUDA DEVICE code starts here 
//==============================================================================

//------------------------------------------------------------------------------
__constant__ SolverConfiguration gConfiguration;    // current solver's config.
//------------------------------------------------------------------------------
enum
{
    // particle state
    PS_DEFAULT = 0,
    PS_DELETE,          // particle is marked for deletion
    PS_INSERT           // particle is marked for insertion
};
//------------------------------------------------------------------------------
#define BLOCK_DIMENSIONS_X 256
#define EMPTY_CELL_ID 0xFFFFFFFF
//------------------------------------------------------------------------------

//==============================================================================
// UTLITY device kernels definition
//==============================================================================

//------------------------------------------------------------------------------
__device__ void computeCoordinatesOff (
    int3& coordinate,            // out: coordinate for [position]
    float3 position,
    const Grid& grid,
    float offset
)
{
    // compute the coordinates of a point in space with respect to the given 
    // grid

    coordinate.x = (int)((position.x + offset - grid.Origin.x)/grid.Spacing);
    coordinate.y = (int)((position.y + offset - grid.Origin.y)/grid.Spacing);
    coordinate.z = (int)((position.z + offset - grid.Origin.z)/grid.Spacing);

    // clamp coordinates if neccessary
    coordinate.x = max(0, min(coordinate.x, grid.Dimensions.x - 1));
    coordinate.y = max(0, min(coordinate.y, grid.Dimensions.y - 1));
    coordinate.z = max(0, min(coordinate.z, grid.Dimensions.z - 1));
}
//------------------------------------------------------------------------------
__device__ void computeCoordinates (
    int3& coordinate,            // out: coordinate for [position]
    float3 position,
    const Grid& grid
)
{
    // compute the coordinates of a point in space with respect to the given 
    // grid

    coordinate.x = (int)((position.x - grid.Origin.x)/grid.Spacing);
    coordinate.y = (int)((position.y - grid.Origin.y)/grid.Spacing);
    coordinate.z = (int)((position.z - grid.Origin.z)/grid.Spacing);

    // clamp coordinates if neccessary
    coordinate.x = max(0, min(coordinate.x, grid.Dimensions.x - 1));
    coordinate.y = max(0, min(coordinate.y, grid.Dimensions.y - 1));
    coordinate.z = max(0, min(coordinate.z, grid.Dimensions.z - 1));
}
//------------------------------------------------------------------------------
__device__ void computeHash (
    unsigned int& hash, 
    const int3& coordinate,
    const Grid& grid
)
{
    // compute the hash for a grid given a coordinate within the grid

    hash = coordinate.x + grid.Dimensions.x*
        (coordinate.y + grid.Dimensions.y*coordinate.z);
}
//------------------------------------------------------------------------------
__device__ void computeHash (
    unsigned int& hash, 
    float3 position,
    const Grid& grid
)
{
    // compute the hash for a grid given a position in world space, by first
    // conputing the coordinate in [grid], and then computing the hash.

    int3 coordinate;
    computeCoordinates(coordinate, position, grid);
    computeHash(hash, coordinate, grid);
}
//------------------------------------------------------------------------------
__device__ inline void computeNorm (float& norm, const float3& a)
{
    norm = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
//------------------------------------------------------------------------------
__device__ inline void computeDistance (
    float& dist, 
    const float3& a, 
    const float3& b
)
{
    float3 d;
    d.x = a.x - b.x;
    d.y = a.y - b.y;
    d.z = a.z - b.z;
    computeNorm(dist, d); 
}
//------------------------------------------------------------------------------
__device__ inline void evaluatePoly6Kernel (
    float& res,  // [out] result of evaluation
    float d,     // distance between two particles
    float h      // effective radius 
)
{
    // evaluate Muellers Poly6 Kernel

    float hhh = h*h*h;
    float coeff = 315.0f/(64.0f*M_PI*hhh*hhh*hhh);

    if (d < h)
    {
        float a = h*h - d*d;
        res = coeff*a*a*a;
    }
    else
    {
        res = 0.0f;    
    }
}
//------------------------------------------------------------------------------
__device__ inline void evaluateSpikyKernelGradient (
    float3& grad,
    const float3& xij,
    float h
)
{
    float norm = 0.0f;
    computeNorm(norm, xij);
    
    if ((norm == 0.0f) || (norm > h))
    {
        grad.x = 0.0f;
        grad.y = 0.0f;
        grad.z = 0.0f;
        return;
    } 

    float hhh = h*h*h;
    float a = -45.0f/(M_PI*hhh*hhh)*(h - norm)*(h - norm);

    grad.x = a*xij.x/norm;
    grad.y = a*xij.y/norm;
    grad.z = a*xij.z/norm;
}
//------------------------------------------------------------------------------
__device__ inline void evaluateViscosityKernelLaplacian (
    float& lapl,
    float dist,
    float h
)
{
    if (dist < h)
    {
        float hhh = h*h*h;
        float coeff = 45.0f/(M_PI*hhh*hhh);
        lapl = coeff*(h - dist);
        return;
    }
    else
    {
        return;
    }
}
//------------------------------------------------------------------------------
__device__ inline void evaluateBoundaryWeight (
    float& weight, 
    float dist, 
    float h
)
{
    float q = 2.0f*dist/h;
    float coeff = 0.02f*gConfiguration.SpeedSound*gConfiguration.SpeedSound/
        dist;

    if (q < 2.0f/3.0f)
    {
        weight = coeff*2.0f/3.0f;
    } 
    else if (q < 1.0f)
    {
        weight = coeff*(2.0f*q - 3.0f/2.0f*q*q);
    }
    else if (q < 2.0f)
    {
        float a = 2.0f - q;
        weight = coeff*0.5f*a*a;
    }
    else
    {
        weight = 0.0f;
    }
}
//------------------------------------------------------------------------------
__device__ inline void computeDensityCell (
    float& rhoi,                 // [out] density of particle i 
    const float3& xi,            // position of particle i
    const float* dPositions,     
    unsigned int start,
    unsigned int end,
    float effectiveRadius
)
{
    // add up density contribution form particle in this cell ([start], [end])
    // to the density of the particle i [rhoi]. (in fact only the kernel 
    // weights are added up, mass is multiplied in the callee, to safe
    // operations)
    
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float dist;
        computeDistance(dist, xi, xj);

        if (dist < effectiveRadius)
        {
            float weight = 0.0f;
            evaluatePoly6Kernel(
                weight, 
                dist, 
                effectiveRadius
            );
            rhoi += weight;
        }
            
    }
  
}
//------------------------------------------------------------------------------
__device__ inline void computeDensityCellComplement (
    float& rhoi,                 // [out] density of particle i 
    const float3& xi,            // position of particle i
    const float* dPositions,     // high resolution positions
    unsigned int start,
    unsigned int end
)
{
    // add up density contribution form particle in this cell ([start], [end])
    // to the density of the particle i [rhoi]. (in fact only the kernel 
    // weights are added up, mass is multiplied in the callee, to safe
    // operations)
    
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float dist;
        computeDistance(dist, xi, xj);

        if (dist < gConfiguration.EffectiveRadius[0])
        {
            float weight0 = 0.0f;
            float weight1 = 0.0f;

            evaluatePoly6Kernel(
                weight0, 
                dist, 
                gConfiguration.EffectiveRadius[0]
            );
            evaluatePoly6Kernel(
                weight1, 
                dist, 
                gConfiguration.EffectiveRadius[1]
            );
            rhoi += 0.5f*(weight0 + weight1);
        }
            
    }
  
}
//------------------------------------------------------------------------------
__device__ inline void computeAccelerationCell (
    float3& fi,
    float rhoi,
    float pi,
    const float3& xi,        
    const float3& vi,    
    const float* dDensities,
    const float* dPressures,
    const float* dPositions,     
    const float* dVelocities,
    unsigned int start,
    unsigned int end,
    unsigned char res           // res of the particle 0 == low, 1 == high
)
{
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float3 vj;
        vj.x = dVelocities[3*j + 0];
        vj.y = dVelocities[3*j + 1];
        vj.z = dVelocities[3*j + 2];
        float rhoj = dDensities[j];
        float pj = dPressures[j];
        float dist;
        float3 xij;
        xij.x = xi.x - xj.x; 
        xij.y = xi.y - xj.y; 
        xij.z = xi.z - xj.z; 
        computeNorm(dist, xij);
        
        if (dist != 0.0f && dist < gConfiguration.EffectiveRadius[res])
        {
            // evaluate the pressure force partice j exerts on particle i
            float coeffP = -rhoi*gConfiguration.FluidParticleMass[res]*
                (pi/(rhoi*rhoi) + pj/(rhoj*rhoj));
            float3 grad;
            evaluateSpikyKernelGradient(
                grad, 
                xij,
                gConfiguration.EffectiveRadius[res]
            );
            fi.x += coeffP*grad.x;
            fi.y += coeffP*grad.y;
            fi.z += coeffP*grad.z;

            // evaluate the viscosity force partice j exerts on particle i
            float coeffV = gConfiguration.Viscosity*
                gConfiguration.FluidParticleMass[res]/rhoj;
            float lapl = 0.0f;
            evaluateViscosityKernelLaplacian(
                lapl, 
                dist, 
                gConfiguration.EffectiveRadius[res]
            );
            float3 vji;
            vji.x = vj.x - vi.x;
            vji.y = vj.y - vi.y;
            vji.z = vj.z - vi.z;
            fi.x += coeffV*vji.x*lapl;
            fi.y += coeffV*vji.y*lapl;
            fi.z += coeffV*vji.z*lapl;

            // evaluate the surface tension force partice j exerts on particle i
            float weight;
            evaluatePoly6Kernel(
                weight, 
                dist, 
                gConfiguration.EffectiveRadius[res]
            );
            float coeffT = -weight*gConfiguration.FluidParticleMass[res]*
                gConfiguration.TensionCoefficient;
        
            fi.x += coeffT*xij.x;
            fi.y += coeffT*xij.y;
            fi.z += coeffT*xij.z;
        }

    }

}
//------------------------------------------------------------------------------
__device__ inline void computeAccelerationCellComplement (
    float3& fi,
    float rhoi,
    float pi,
    const float3& xi,        
    const float3& vi,    
    const float* dDensities,
    const float* dPressures,
    const float* dPositions,     
    const float* dVelocities,
    unsigned int start,
    unsigned int end,
    unsigned char res           // res of the particle 0 == low, 1 == high
)
{
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float3 vj;
        vj.x = dVelocities[3*j + 0];
        vj.y = dVelocities[3*j + 1];
        vj.z = dVelocities[3*j + 2];
        float rhoj = dDensities[j];
        float pj = dPressures[j];
        float dist;
        float3 xij;
        xij.x = xi.x - xj.x; 
        xij.y = xi.y - xj.y; 
        xij.z = xi.z - xj.z; 
        computeNorm(dist, xij);
        
        if (dist != 0.0f && dist < gConfiguration.EffectiveRadius[0])
        {
            // evaluate the pressure force partice j exerts on particle i
            float coeffP = -rhoi*gConfiguration.FluidParticleMass[res]*
                (pi/(rhoi*rhoi) + pj/(rhoj*rhoj));
            float3 grad0;
            float3 grad1;
            evaluateSpikyKernelGradient(
                grad0, 
                xij,
                gConfiguration.EffectiveRadius[res]
            );
            evaluateSpikyKernelGradient(
                grad1, 
                xij,
                gConfiguration.EffectiveRadius[res]
            );
            fi.x += coeffP*0.5f*(grad0.x + grad1.x);
            fi.y += coeffP*0.5f*(grad0.y + grad1.y);
            fi.z += coeffP*0.5f*(grad0.z + grad1.z);

            // evaluate the viscosity force partice j exerts on particle i
            float coeffV = gConfiguration.Viscosity*
                gConfiguration.FluidParticleMass[res]/rhoj;
            float lapl0 = 0.0f;
            float lapl1 = 0.0f;
            evaluateViscosityKernelLaplacian(
                lapl0, 
                dist, 
                gConfiguration.EffectiveRadius[res]
            );
            evaluateViscosityKernelLaplacian(
                lapl1, 
                dist, 
                gConfiguration.EffectiveRadius[res]
            );
            float3 vji;
            vji.x = vj.x - vi.x;
            vji.y = vj.y - vi.y;
            vji.z = vj.z - vi.z;
            fi.x += coeffV*vji.x*0.5f*(lapl0 + lapl1);
            fi.y += coeffV*vji.y*0.5f*(lapl0 + lapl1);
            fi.z += coeffV*vji.z*0.5f*(lapl0 + lapl1);

            // evaluate the surface tension force partice j exerts on particle i
            float weight0 = 0.0f;
            float weight1 = 0.0f;
            evaluatePoly6Kernel(
                weight0, 
                dist, 
                gConfiguration.EffectiveRadius[0]
            );
            evaluatePoly6Kernel(
                weight1, 
                dist, 
                gConfiguration.EffectiveRadius[1]
            );
            float coeffT = -0.5f*(weight0 + weight1)*
                gConfiguration.FluidParticleMass[res]*
                gConfiguration.TensionCoefficient;
        
            fi.x += coeffT*xij.x;
            fi.y += coeffT*xij.y;
            fi.z += coeffT*xij.z;
        }

    }

}
//------------------------------------------------------------------------------
__device__ void computeBoundaryForceCell (
    float3& bi,
    const float3& xi,
    const float* dPositions,     
    unsigned int start,
    unsigned int end,
    unsigned char res
)
{
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float3 xij;
        xij.x = xi.x - xj.x;
        xij.y = xi.y - xj.y;
        xij.z = xi.z - xj.z;
        float dist;
        computeNorm(dist, xij); 

        if (dist < gConfiguration.EffectiveRadius[res])
        {
            float weight = 0.0f;
            evaluateBoundaryWeight(
                weight, 
                dist, 
                gConfiguration.EffectiveRadius[res]
            );
            weight*= gConfiguration.BoundaryParticleMass/
                (gConfiguration.FluidParticleMass[res] + 
                gConfiguration.BoundaryParticleMass);
            bi.x += weight*xij.x/dist;
            bi.y += weight*xij.y/dist;
            bi.z += weight*xij.z/dist;
        }
    }
}
//------------------------------------------------------------------------------

//==============================================================================
// GLOBAL device kernel definitions
//==============================================================================

//------------------------------------------------------------------------------
__global__ void computeHashsLowD
(
    unsigned int* dHashs,           // hash values of each particle
    unsigned int* dActiveIDs,       // array of active particle ids
    const float* dPositions,        // positions of each particle 
    unsigned int numParticles       // number of ids in the id array
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dActiveIDs[idx];

    float3 pos;
    pos.x = dPositions[3*id + 0];
    pos.y = dPositions[3*id + 1];
    pos.z = dPositions[3*id + 2];

    computeHash(dHashs[idx], pos, gConfiguration.Grid[0]);
};
//------------------------------------------------------------------------------
__global__ void computeHashsHighD
(
    unsigned int* dHashs,           // hash values of each particle
    unsigned int* dActiveIDs,       // array of active particle ids
    const float* dPositions,        // positions of each particle 
    unsigned int numParticles       // number of ids in the id array
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    unsigned int id = dActiveIDs[idx];

    float3 pos;
    pos.x = dPositions[3*id + 0];
    pos.y = dPositions[3*id + 1];
    pos.z = dPositions[3*id + 2];

    computeHash(dHashs[idx], pos, gConfiguration.Grid[1]);
};
//------------------------------------------------------------------------------
__global__ void reorderComputeCellStartEndBoundaryD
(
    unsigned int* dCellStart,
    unsigned int* dCellEnd,
    float* dTempPositions,
    const float* dPositions,
    const unsigned int* dSortedIDs,
    const unsigned int* dHashs,
    unsigned int numParticles
)
{
    extern __shared__ int sharedHash[];
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles) 
    {
        return;
    }

    // reorder
    unsigned int id = dSortedIDs[idx];
    dTempPositions[3*idx + 0] = dPositions[3*id + 0];
    dTempPositions[3*idx + 1] = dPositions[3*id + 1];
    dTempPositions[3*idx + 2] = dPositions[3*id + 2];

    // compute cell start end
    int hash = dHashs[idx];
    sharedHash[threadIdx.x + 1] = hash;
        
    if (idx > 0 && threadIdx.x == 0) 
    {
        sharedHash[0] = dHashs[idx - 1];
    }

    __syncthreads();

    if (idx == 0 || hash != sharedHash[threadIdx.x])
    {
        dCellStart[hash] = idx;
        
        if (idx > 0) 
        {
            dCellEnd[sharedHash[threadIdx.x]] = idx;
        }
    }

    if (idx == numParticles - 1)
    {
        dCellEnd[hash] = idx + 1;
    }
}
//------------------------------------------------------------------------------
__global__ void reorderAndComputeCellStartEndD
(
    unsigned int* dCellStart,
    unsigned int* dCellEnd,
    float* dTempPositions,
    float* dTempVelocities,
    float* dTempBlendCoefficients,
    unsigned char* dTempStates,
    unsigned int* dSortedIDs,
    const float* dPositions,
    const float* dVelocities,
    const float* dBlendVelocities,
    const unsigned char* dStates,
    const unsigned int* dHashs,
    unsigned int numParticles
)
{
    extern __shared__ int sharedHash[];
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles) 
    {
        return;
    }
    
    // reorder
    unsigned int id = dSortedIDs[idx];
    dTempPositions[3*idx + 0] = dPositions[3*id + 0]; 
    dTempPositions[3*idx + 1] = dPositions[3*id + 1]; 
    dTempPositions[3*idx + 2] = dPositions[3*id + 2]; 
    dTempVelocities[3*idx + 0] = dVelocities[3*id + 0]; 
    dTempVelocities[3*idx + 1] = dVelocities[3*id + 1]; 
    dTempVelocities[3*idx + 2] = dVelocities[3*id + 2];
    dTempBlendCoefficients[idx] = dBlendVelocities[id];
    dTempStates[idx] = dStates[id];
    dSortedIDs[idx] = idx;

    // compute cell start end ids
    int hash = dHashs[idx];
    sharedHash[threadIdx.x + 1] = hash;
        
    if (idx > 0 && threadIdx.x == 0) 
    {
        sharedHash[0] = dHashs[idx - 1];
    }

    __syncthreads();

    if (idx == 0 || hash != sharedHash[threadIdx.x])
    {
        dCellStart[hash] = idx;
        
        if (idx > 0) 
        {
            dCellEnd[sharedHash[threadIdx.x]] = idx;
        }
    }

    if (idx == numParticles - 1)
    {
        dCellEnd[hash] = idx + 1;
    }
}
//------------------------------------------------------------------------------
__global__ void computeDensitiesPressuresLowD (
    float* dDensities,              // [out] computed densities
    float* dPressures,              // [out] computed pressures
    const float* dPositions,
    const float* dPositionsHigh,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const unsigned int* dCellStartHigh,
    const unsigned int* dCellEndHigh,
    unsigned int numParticles
)
{
    // compute densities and pressures for the low resolution particles.

    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    float3 xi;
    xi.x = dPositions[3*idx + 0];
    xi.y = dPositions[3*idx + 1];
    xi.z = dPositions[3*idx + 2];

    float rhoi = 0.0f;
    float rhoiComplement = 0.0f;
    int3 cs, ce;
    int3 cc;

    //--------------------------------------------------------------------------
    // compute the density contribution of the low res domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[0], 
        -gConfiguration.EffectiveRadius[0]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[0], 
        gConfiguration.EffectiveRadius[0]
    );


    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[0]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeDensityCell(
                    rhoi,
                    xi,
                    dPositions,
                    start,
                    end,
                    gConfiguration.EffectiveRadius[0]
                );
            }
        }
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // compute the density contribution of the high res domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[1], 
        -gConfiguration.EffectiveRadius[0]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[1], 
        gConfiguration.EffectiveRadius[0]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[1]);
                unsigned int start = dCellStartHigh[hash];
                unsigned int end = dCellStartHigh[hash];

                computeDensityCellComplement(
                    rhoiComplement,
                    xi,
                    dPositionsHigh,
                    start,
                    end
                );
            }
        }
    }
    //--------------------------------------------------------------------------    

    rhoi = rhoi*gConfiguration.FluidParticleMass[0] + 
        rhoiComplement*gConfiguration.FluidParticleMass[1];
    dDensities[idx] = rhoi;
    dPressures[idx] = gConfiguration.BulkModulus*
        (rhoi - gConfiguration.RestDensity);
}
//------------------------------------------------------------------------------
__global__ void computeDensitiesPressuresHighD (
    float* dDensities,              // [out] computed densities
    float* dPressures,              // [out] computed pressures
    const float* dPositions,
    const float* dPositionsLow,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const unsigned int* dCellStartLow,
    const unsigned int* dCellEndLow,
    unsigned int numParticles
)
{
    // compute densities and pressures for the high resolution particles.

    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    float3 xi;
    xi.x = dPositions[3*idx + 0];
    xi.y = dPositions[3*idx + 1];
    xi.z = dPositions[3*idx + 2];

    float rhoi = 0.0f;
    float rhoiComplement = 0.0f;
    int3 cs, ce;
    int3 cc;

    //--------------------------------------------------------------------------
    // compute the density contribution of the high res domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[1], 
        -gConfiguration.EffectiveRadius[1]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[1], 
        gConfiguration.EffectiveRadius[1]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[1]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeDensityCell(
                    rhoi,
                    xi,
                    dPositions,
                    start,
                    end,
                    gConfiguration.EffectiveRadius[1]
                );

                rhoi += end-EMPTY_CELL_ID;
            }
        }
    }
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // compute the density contribution of the low res domain
    //computeCoordinatesOff(
    //    cs, 
    //    xi, 
    //    gConfiguration.Grid[0], 
    //    -gConfiguration.EffectiveRadius[0]
    //);
    //computeCoordinatesOff(
    //    ce, 
    //    xi, 
    //    gConfiguration.Grid[0], 
    //    gConfiguration.EffectiveRadius[0]
    //);

    //for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    //{
    //    for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
    //    {
    //        for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
    //        {
    //            unsigned int hash;
    //            computeHash(hash, cc, gConfiguration.Grid[0]);
    //            unsigned int start = dCellStartLow[hash];
    //            unsigned int end = dCellStartLow[hash];

    //            computeDensityCellComplement(
    //                rhoiComplement,
    //                xi,
    //                dPositionsLow,
    //                start,
    //                end
    //            );
    //        }
    //    }
    //}
    //--------------------------------------------------------------------------    

    rhoi = rhoi*gConfiguration.FluidParticleMass[1] + 
        rhoiComplement*gConfiguration.FluidParticleMass[0];
    dDensities[idx] = rhoi;
    dPressures[idx] = gConfiguration.BulkModulus*
        (rhoi - gConfiguration.RestDensity);
}
//------------------------------------------------------------------------------
__global__ void computeAccelerationsAndUpdateStatesLowD (
    float* dAccelerations,
    unsigned char* dStates,
    const float* dDensities,              
    const float* dPressures,
    const float* dPositions,
    const float* dVelocities,
    const float* dDensitiesHigh,              
    const float* dPressuresHigh,
    const float* dPositionsHigh,
    const float* dVelocitiesHigh,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const unsigned int* dCellStartHigh,
    const unsigned int* dCellEndHigh,
    const float* dBoundaryPositions,
    const unsigned int* dBoundaryCellStart,
    const unsigned int* dBoundaryCellEnd,
    unsigned int numParticles
)
{
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    float3 xi;
    xi.x = dPositions[3*idx + 0];
    xi.y = dPositions[3*idx + 1];
    xi.z = dPositions[3*idx + 2];
    float3 vi;
    vi.x = dVelocities[3*idx + 0];
    vi.y = dVelocities[3*idx + 1];
    vi.z = dVelocities[3*idx + 2];
    float rhoi = dDensities[idx];
    float pi = dPressures[idx];
    float3 fi;
    fi.x = 0.0;
    fi.y = 0.0;
    fi.z = 0.0;
    float3 bi;
    bi.x = 0.0f;
    bi.y = 0.0f;
    bi.z = 0.0f;
    int3 cs, ce;
    int3 cc;

    // compute force contribution from the same domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[0], 
        -gConfiguration.EffectiveRadius[0]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[0], 
        gConfiguration.EffectiveRadius[0]
    );
    
    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[0]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeAccelerationCell(
                    fi,
                    rhoi,
                    pi,
                    xi,
                    vi,
                    dDensities,
                    dPressures,
                    dPositions,
                    dVelocities,
                    start,
                    end,
                    0
                );
            }
        }
    }

    // compute force contribution from the complement domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[1], 
        -gConfiguration.EffectiveRadius[1]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[1], 
        gConfiguration.EffectiveRadius[1]
    );
    
    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[1]);
                unsigned int start = dCellStartHigh[hash];
                unsigned int end = dCellEndHigh[hash];

                computeAccelerationCellComplement(
                    fi,
                    rhoi,
                    pi,
                    xi,
                    vi,
                    dDensitiesHigh,
                    dPressuresHigh,
                    dPositionsHigh,
                    dVelocitiesHigh,
                    start,
                    end,
                    1
                );
            }
        }
    }

    // compute force contribution from the boundary
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[0], 
        -gConfiguration.EffectiveRadius[0]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[0], 
        gConfiguration.EffectiveRadius[0]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[0]);
                unsigned int start = dBoundaryCellStart[hash];
                unsigned int end = dBoundaryCellEnd[hash];
                computeBoundaryForceCell(
                    bi,
                    xi,
                    dBoundaryPositions,
                    start,
                    end,
                    0
                );
            }
        }
    }

    dAccelerations[3*idx + 0] = fi.x/rhoi + bi.x;
    dAccelerations[3*idx + 1] = fi.y/rhoi - 9.81f + bi.y;
    dAccelerations[3*idx + 2] = fi.z/rhoi + bi.z;

    // update states
    //if (xi.x > 0.5f)
    //{
    //    dStates[idx] = PS_DELETE;
    //}
}
//------------------------------------------------------------------------------
__global__ void computeAccelerationsAndUpdateStatesHighD (
    float* dAccelerations,
    unsigned char* dStates,
    const float* dDensities,              
    const float* dPressures,
    const float* dPositions,
    const float* dVelocities,
    const float* dDensitiesLow,              
    const float* dPressuresLow,
    const float* dPositionsLow,
    const float* dVelocitiesLow,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const unsigned int* dCellStartLow,
    const unsigned int* dCellEndLow,
    const float* dBoundaryPositions,
    const unsigned int* dBoundaryCellStart,
    const unsigned int* dBoundaryCellEnd,
    unsigned int numParticles
)
{
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    float3 xi;
    xi.x = dPositions[3*idx + 0];
    xi.y = dPositions[3*idx + 1];
    xi.z = dPositions[3*idx + 2];
    float3 vi;
    vi.x = dVelocities[3*idx + 0];
    vi.y = dVelocities[3*idx + 1];
    vi.z = dVelocities[3*idx + 2];
    float rhoi = dDensities[idx];
    float pi = dPressures[idx];
    float3 fi;
    fi.x = 0.0;
    fi.y = 0.0;
    fi.z = 0.0;
    float3 bi;
    bi.x = 0.0f;
    bi.y = 0.0f;
    bi.z = 0.0f;
    int3 cs, ce;
    int3 cc;

    // compute force contribution from the same domain
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[1], 
        -gConfiguration.EffectiveRadius[1]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[1], 
        gConfiguration.EffectiveRadius[1]
    );
    
    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[1]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeAccelerationCell(
                    fi,
                    rhoi,
                    pi,
                    xi,
                    vi,
                    dDensities,
                    dPressures,
                    dPositions,
                    dVelocities,
                    start,
                    end,
                    1
                );
            }
        }
    }

    // compute force contribution from the complement domain
    //computeCoordinatesOff(
    //    cs, 
    //    xi, 
    //    gConfiguration.Grid[0], 
    //    -gConfiguration.EffectiveRadius[0]
    //);
    //computeCoordinatesOff(
    //    ce, 
    //    xi, 
    //    gConfiguration.Grid[0], 
    //    gConfiguration.EffectiveRadius[0]
    //);
    //
    //for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    //{
    //    for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
    //    {
    //        for (cc.x = cs.x; cc.x <= ce.x; cc.x++)
    //        {
    //            unsigned int hash;
    //            computeHash(hash, cc, gConfiguration.Grid[0]);
    //            unsigned int start = dCellStartLow[hash];
    //            unsigned int end = dCellEndLow[hash];

    //            computeAccelerationCellComplement(
    //                fi,
    //                rhoi,
    //                pi,
    //                xi,
    //                vi,
    //                dDensitiesLow,
    //                dPressuresLow,
    //                dPositionsLow,
    //                dVelocitiesLow,
    //                start,
    //                end,
    //                0
    //            );
    //        }
    //    }
    //}

    // compute force contribution from the boundary
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[0], 
        -gConfiguration.EffectiveRadius[0]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[0], 
        gConfiguration.EffectiveRadius[0]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[0]);
                unsigned int start = dBoundaryCellStart[hash];
                unsigned int end = dBoundaryCellEnd[hash];
                computeBoundaryForceCell(
                    bi,
                    xi,
                    dBoundaryPositions,
                    start,
                    end,
                    1
                );
            }
        }
    }

    dAccelerations[3*idx + 0] = fi.x/rhoi + bi.x;
    dAccelerations[3*idx + 1] = fi.y/rhoi - 9.81f + bi.y;
    dAccelerations[3*idx + 2] = fi.z/rhoi + bi.z;
}
//------------------------------------------------------------------------------
__global__ void updateSystemD (
    float* dPositions, 
    float* dVelocities, 
    float* dBlendCoefficients,
    unsigned char* dStates,
    unsigned int* dActiveIDs,
    unsigned int* dNumParticles,
    const float* dTempPositions,
    const float* dTempVelocities,
    const float* dTempBlendCoefficients,
    const unsigned char* dTempStates,
    const float* dAccelerations,
    float timeStep,
    unsigned int numParticles
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    float3 xi;
    xi.x = dTempPositions[3*idx + 0];
    xi.y = dTempPositions[3*idx + 1];
    xi.z = dTempPositions[3*idx + 2];
    float3 vi;
    vi.x = dTempVelocities[3*idx + 0];
    vi.y = dTempVelocities[3*idx + 1];
    vi.z = dTempVelocities[3*idx + 2];

    // update position and velocity of the particle
    vi.x += timeStep*dAccelerations[3*idx + 0]; 
    vi.y += timeStep*dAccelerations[3*idx + 1]; 
    vi.z += timeStep*dAccelerations[3*idx + 2]; 
    xi.x += timeStep*vi.x;
    xi.y += timeStep*vi.y;
    xi.z += timeStep*vi.z;

    // store new position and velocity of the particle
    dPositions[3*idx + 0] = xi.x;
    dPositions[3*idx + 1] = xi.y;
    dPositions[3*idx + 2] = xi.z;

    dVelocities[3*idx + 0] = vi.x;
    dVelocities[3*idx + 1] = vi.y;
    dVelocities[3*idx + 2] = vi.z;

    // update blend values

    // [c] is a mapping functions, since for default particles blendvalues
    // should not be changed (i.e. 0 => 0.0f), for particles for deletion
    // the blend values should be substracted by the blend increment 
    // (i.e. 1 => -1.0), for particles marked for insertion blend values
    // should be added by the blend increment (i.e. 2 => 1.0f)
    const float c[] = {0.0f, -1.0f, 1.0f};

    float blendVal = dTempBlendCoefficients[idx];
    unsigned char state = dTempStates[idx];

    blendVal += c[state]*gConfiguration.BlendIncrement;

    dBlendCoefficients[idx] = blendVal;
    dStates[idx] = state;

    // if the blend value of the particle is greater than zero (i.e. it 
    // contributes to the system) add it to the particle list for computations
    // in the next time step. Otherwise, disregard the particle.
    if (blendVal > 0.0f)
    {
        // (atomic) add one to the particle count
        unsigned int old = atomicAdd(dNumParticles, 1);
        dActiveIDs[old] = idx;
    }
}
//------------------------------------------------------------------------------

//==============================================================================
//  HOST code starts here 
//==============================================================================

//==============================================================================
//  UTILITY functions start here
//==============================================================================

//------------------------------------------------------------------------------
void computeGridDimensions 
(
    dim3& gridDimensions, 
    const dim3& blockDimensions,
    unsigned int numParticles
)
{
    // compute the dimensions of the cuda grid for tgiven block dimensions,
    // and the number of particles

    if (numParticles % blockDimensions.x == 0)
    {
        gridDimensions.x = numParticles/blockDimensions.x;
    }
    else
    {
        gridDimensions.x = numParticles/blockDimensions.x + 1;    
    }
    gridDimensions.y = 1;
    gridDimensions.z = 1;
}
//------------------------------------------------------------------------------

//==============================================================================
// SPHParticleData's definition
//==============================================================================

//------------------------------------------------------------------------------
Solver::SPHParticleData::SPHParticleData (
    ParticleData* data, 
    unsigned int numGridCells   // # of grid cells in each direction
)
: 
    Data(data),
    BlockDimensions(BLOCK_DIMENSIONS_X, 1, 1), 
    NumGridCells(numGridCells)
{
    // allocate additional memory for storing density, pressure, acceleration,
    // velocities and hash values for the particles and initialize that data.
    // also allocate memory for the neighbor search as described in the nVidia
    // particles white paper
    CUDA::Alloc<float>(&dDensities, data->MaxParticles);
    CUDA::Alloc<float>(&dPressures, data->MaxParticles);
    CUDA::Alloc<float>(&dAccelerations, 3*data->MaxParticles);
    CUDA::Alloc<float>(&dVelocities, 3*data->MaxParticles);
    CUDA::Alloc<float>(&dTempPositions, 3*data->MaxParticles);
    CUDA::Alloc<float>(&dTempVelocities, 3*data->MaxParticles);
    CUDA::Alloc<float>(&dBlendCoefficients, data->MaxParticles);
    CUDA::Alloc<float>(&dTempBlendCoefficients, data->MaxParticles);
    CUDA::Alloc<unsigned char>(&dStates, data->MaxParticles);
    CUDA::Alloc<unsigned char>(&dTempStates, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dNumParticles, 1);
    CUDA::Fill<float>(dDensities, data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dPressures, data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dAccelerations, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dVelocities, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dTempPositions, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dTempVelocities, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dBlendCoefficients, data->MaxParticles, 1.0f);
    CUDA::Fill<float>(dTempBlendCoefficients, data->MaxParticles, 1.0f);
    CUDA::Fill<unsigned char>(dStates, data->MaxParticles, PS_DEFAULT);
    CUDA::Fill<unsigned char>(dTempStates, data->MaxParticles, PS_DEFAULT);
    CUDA::Memcpy<unsigned int>(
        dNumParticles, 
        &Data->NumParticles, 
        1, 
        cudaMemcpyHostToDevice
    );
    CUDA::Alloc<unsigned int>(&dActiveIDs, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dHashs, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dCellStart, numGridCells);
    CUDA::Alloc<unsigned int>(&dCellEnd, numGridCells);
    CUDA::Fill<unsigned int>(dActiveIDs, data->MaxParticles, 0, 1);
    CUDA::Fill<unsigned int>(dHashs, data->MaxParticles, 0);
    CUDA::Fill<unsigned int>(dCellStart, numGridCells, 0);
    CUDA::Fill<unsigned int>(dCellEnd, numGridCells, 0);

    // compute the number of cuda blocks we need based on the current number
    // of particles and the threads per block we use. Also compute the amount
    // of shared memory we need to compute the values for [dCellStart] and
    // [dCellEnd]
    computeGridDimensions(GridDimensions, BlockDimensions, data->NumParticles);
    SharedMemSize = sizeof(int)*(BlockDimensions.x + 1);
}
//------------------------------------------------------------------------------
Solver::SPHParticleData::~SPHParticleData ()
{
    // free everything

    CUDA::Free<float>(&dDensities);
    CUDA::Free<float>(&dAccelerations);
    CUDA::Free<float>(&dPressures);
    CUDA::Free<float>(&dVelocities);
    CUDA::Free<float>(&dTempVelocities);
    CUDA::Free<float>(&dTempPositions);
    CUDA::Free<float>(&dBlendCoefficients);
    CUDA::Free<float>(&dTempBlendCoefficients);
    CUDA::Free<unsigned char>(&dStates);
    CUDA::Free<unsigned char>(&dTempStates);
    CUDA::Free<unsigned int>(&dHashs);
    CUDA::Free<unsigned int>(&dCellStart);
    CUDA::Free<unsigned int>(&dCellEnd);
}
//------------------------------------------------------------------------------
Solver::BoundaryParticleData::BoundaryParticleData (
    ParticleData* data, 
    unsigned int numGridCells
)
:
    Data(data),
    BlockDimensions(BLOCK_DIMENSIONS_X, 1, 1), 
    NumGridCells(numGridCells)
{
    CUDA::Alloc<unsigned int>(&dHashs, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dCellStart, numGridCells);
    CUDA::Alloc<unsigned int>(&dCellEnd, numGridCells);
    CUDA::Fill<unsigned int>(dHashs, data->MaxParticles, 0);
    CUDA::Fill<unsigned int>(dCellStart, numGridCells, 0);
    CUDA::Fill<unsigned int>(dCellEnd, numGridCells, 0);    

    computeGridDimensions(GridDimensions, BlockDimensions, data->NumParticles);
    SharedMemSize = sizeof(int)*(BlockDimensions.x + 1);
}
//------------------------------------------------------------------------------
Solver::BoundaryParticleData::~BoundaryParticleData ()
{
    CUDA::Free<unsigned int>(&dHashs);
    CUDA::Free<unsigned int>(&dCellStart);
    CUDA::Free<unsigned int>(&dCellEnd);
}
//------------------------------------------------------------------------------

//==============================================================================
// Solvers's definition
//==============================================================================

//------------------------------------------------------------------------------
Solver::Solver 
(
    ParticleData* fluidData, 
    ParticleData* fluidDataHigh,
    ParticleData* boundaryData,
    const SolverConfiguration* configuration
)
: 
    mConfiguration(*configuration), 
    mFluidData(
        fluidData, 
        Grid::ComputeNumGridCells(configuration->Grid[0])
    ), 
    mFluidDataHigh(
        fluidDataHigh, 
        Grid::ComputeNumGridCells(configuration->Grid[1])
    ), 
    mBoundaryData(
        boundaryData, 
        Grid::ComputeNumGridCells(configuration->Grid[1])
    )
{
    // store pointer to fluid particles and boundary particles also store
    // a copy of solver configuration

    //--------------------------------------------------------------------------
    // compute neighborhood of boundary particles beforehand 
    unsigned int* dBoundaryIDs;
    CUDA::Alloc<unsigned int>(&dBoundaryIDs, mBoundaryData.Data->MaxParticles);
    CUDA::Fill<unsigned int>(dBoundaryIDs, mBoundaryData.Data->MaxParticles, 
        0, 1);

    float* dBoundaryPositions;
    CUDA::Alloc<float>(&dBoundaryPositions, 3*mBoundaryData.Data->MaxParticles);
    
    this->Bind();   // bind first
    mBoundaryData.Data->Map();
    computeHashsLowD<<<mBoundaryData.GridDimensions, 
        mBoundaryData.BlockDimensions>>>(
        mBoundaryData.dHashs, 
        dBoundaryIDs, 
        mBoundaryData.Data->dPositions, 
        mBoundaryData.Data->NumParticles
    );
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(mBoundaryData.dHashs),
        thrust::device_ptr<unsigned int>(mBoundaryData.dHashs + 
            mBoundaryData.Data->NumParticles),
        thrust::device_ptr<unsigned int>(dBoundaryIDs)
    );
    CUDA::Memset<unsigned int>(
        mBoundaryData.dCellStart, 
        EMPTY_CELL_ID, 
        mBoundaryData.NumGridCells
    );
    CUDA::Memset<unsigned int>(
        mBoundaryData.dCellEnd, 
        EMPTY_CELL_ID, 
        mBoundaryData.NumGridCells
    );
    reorderComputeCellStartEndBoundaryD<<<mBoundaryData.GridDimensions,
        mBoundaryData.BlockDimensions, 
        mBoundaryData.SharedMemSize>>>(
        mBoundaryData.dCellStart,
        mBoundaryData.dCellEnd,
        dBoundaryPositions,
        mBoundaryData.Data->dPositions,
        dBoundaryIDs,
        mBoundaryData.dHashs,
        mBoundaryData.Data->NumParticles 
    );
    CUDA::Memcpy<float>(
        mBoundaryData.Data->dPositions, 
        dBoundaryPositions, 
        3*mBoundaryData.Data->MaxParticles,
        cudaMemcpyDeviceToDevice
    );
    mBoundaryData.Data->Unmap();

    CUDA::Free<unsigned int>(&dBoundaryIDs);
    CUDA::Free<float>(&dBoundaryPositions);
    //--------------------------------------------------------------------------

}
//------------------------------------------------------------------------------
Solver::~Solver ()
{

}
//------------------------------------------------------------------------------
void Solver::Bind () const
{
    // set the configuration of this solver on the device
    CUDA::SafeCall(
        cudaMemcpyToSymbol(
            gConfiguration, 
            &mConfiguration, 
            sizeof(mConfiguration)
        ), 
        __FILE__, 
        __LINE__ 
    );
}
//------------------------------------------------------------------------------
void Solver::Advance (float timeStep)
{
    CUDA::Timer t;
    t.Start();
    mFluidData.Data->Map();
    mFluidDataHigh.Data->Map();
    mBoundaryData.Data->Map();
    this->computeNeighborhoodsLow();
    this->computeNeighborhoodsHigh();
    this->computeDensitiesLow();
    this->computeDensitiesHigh();
    //CUDA::DumpArray<float>(mFluidDataHigh.dDensities, mFluidData.Data->NumParticles);
    this->computeAccelerationsAndUpdateStatesLow();
    this->computeAccelerationsAndUpdateStatesHigh();
    this->updateSystem(mFluidData, 0.0f);
    this->updateSystem(mFluidDataHigh, timeStep);
    mBoundaryData.Data->Unmap();
    mFluidDataHigh.Data->Unmap();
    mFluidData.Data->Unmap();
    t.Stop();
    t.DumpElapsed();
}
//------------------------------------------------------------------------------
void Solver::computeNeighborhoodsLow ()
{    
    // copy back from device the new amount of particles for this time step
    CUDA::Memcpy<unsigned int>(
        &mFluidData.Data->NumParticles,
        mFluidData.dNumParticles,
        1,
        cudaMemcpyDeviceToHost
    );

    // compute hashs of all particles
    computeHashsLowD<<<mFluidData.GridDimensions, mFluidData.BlockDimensions>>>(
        mFluidData.dHashs,
        mFluidData.dActiveIDs,
        mFluidData.Data->dPositions,
        mFluidData.Data->NumParticles
    );
        
    // sort the active particle ids by their hash
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(mFluidData.dHashs),
        thrust::device_ptr<unsigned int>(mFluidData.dHashs + 
            mFluidData.Data->NumParticles),
        thrust::device_ptr<unsigned int>(mFluidData.dActiveIDs)
    );

    // reset, then compute cell start end list
    CUDA::Memset<unsigned int>(
        mFluidData.dCellStart, 
        EMPTY_CELL_ID, 
        mFluidData.NumGridCells
    );
    CUDA::Memset<unsigned int>(
        mFluidData.dCellEnd, 
        EMPTY_CELL_ID, 
        mFluidData.NumGridCells
    );
    reorderAndComputeCellStartEndD<<<mFluidData.GridDimensions, 
        mFluidData.BlockDimensions, mFluidData.SharedMemSize>>>(
        mFluidData.dCellStart,
        mFluidData.dCellEnd,
        mFluidData.dTempPositions,
        mFluidData.dTempVelocities,
        mFluidData.dTempBlendCoefficients,
        mFluidData.dTempStates,
        mFluidData.dActiveIDs,
        mFluidData.Data->dPositions,
        mFluidData.dVelocities,
        mFluidData.dBlendCoefficients,
        mFluidData.dStates,
        mFluidData.dHashs,
        mFluidData.Data->NumParticles
    );

}
//------------------------------------------------------------------------------
void Solver::computeNeighborhoodsHigh ()
{
    // copy back from device the new amount of particles for this time step
    CUDA::Memcpy<unsigned int>(
        &mFluidDataHigh.Data->NumParticles,
        mFluidDataHigh.dNumParticles,
        1,
        cudaMemcpyDeviceToHost
    );

    // compute hashs of all particles
    computeHashsHighD<<<mFluidDataHigh.GridDimensions, 
        mFluidDataHigh.BlockDimensions>>>(
        mFluidDataHigh.dHashs,
        mFluidDataHigh.dActiveIDs,
        mFluidDataHigh.Data->dPositions,
        mFluidDataHigh.Data->NumParticles
    );
        
    // sort the active particle ids by their hash
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(mFluidDataHigh.dHashs),
        thrust::device_ptr<unsigned int>(mFluidDataHigh.dHashs + 
            mFluidDataHigh.Data->NumParticles),
        thrust::device_ptr<unsigned int>(mFluidDataHigh.dActiveIDs)
    );

    // reset, then compute cell start end list
    CUDA::Memset<unsigned int>(
        mFluidDataHigh.dCellStart, 
        EMPTY_CELL_ID, 
        mFluidDataHigh.NumGridCells
    );
    CUDA::Memset<unsigned int>(
        mFluidDataHigh.dCellEnd, 
        EMPTY_CELL_ID, 
        mFluidDataHigh.NumGridCells
    );
    reorderAndComputeCellStartEndD<<<mFluidDataHigh.GridDimensions, 
        mFluidDataHigh.BlockDimensions, mFluidDataHigh.SharedMemSize>>>(
        mFluidDataHigh.dCellStart,
        mFluidDataHigh.dCellEnd,
        mFluidDataHigh.dTempPositions,
        mFluidDataHigh.dTempVelocities,
        mFluidDataHigh.dTempBlendCoefficients,
        mFluidDataHigh.dTempStates,
        mFluidDataHigh.dActiveIDs,
        mFluidDataHigh.Data->dPositions,
        mFluidDataHigh.dVelocities,
        mFluidDataHigh.dBlendCoefficients,
        mFluidDataHigh.dStates,
        mFluidDataHigh.dHashs,
        mFluidDataHigh.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
void Solver::computeDensitiesLow ()
{
    computeDensitiesPressuresLowD<<<mFluidData.GridDimensions, 
        mFluidData.BlockDimensions>>>(
        mFluidData.dDensities,
        mFluidData.dPressures,
        mFluidData.dTempPositions,
        mFluidDataHigh.dTempPositions,
        mFluidData.dCellStart,
        mFluidData.dCellEnd,
        mFluidDataHigh.dCellStart,
        mFluidDataHigh.dCellEnd,
        mFluidData.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
void Solver::computeDensitiesHigh ()
{
    computeDensitiesPressuresHighD<<<mFluidDataHigh.GridDimensions, 
        mFluidDataHigh.BlockDimensions>>>(
        mFluidDataHigh.dDensities,
        mFluidDataHigh.dPressures,
        mFluidDataHigh.dTempPositions,
        mFluidData.dTempPositions,
        mFluidDataHigh.dCellStart,
        mFluidDataHigh.dCellEnd,
        mFluidData.dCellStart,
        mFluidData.dCellEnd,
        mFluidDataHigh.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
void Solver::computeAccelerationsAndUpdateStatesLow ()
{
    // computes the accelerations

    computeAccelerationsAndUpdateStatesLowD<<<mFluidData.GridDimensions, 
        mFluidData.BlockDimensions>>>(
        mFluidData.dAccelerations,
        mFluidData.dTempStates,
        mFluidData.dDensities,
        mFluidData.dPressures,
        mFluidData.dTempPositions,
        mFluidData.dTempVelocities,
        mFluidDataHigh.dDensities,
        mFluidDataHigh.dPressures,
        mFluidDataHigh.dTempPositions,
        mFluidDataHigh.dTempVelocities,
        mFluidData.dCellStart,
        mFluidData.dCellEnd,
        mFluidDataHigh.dCellStart,
        mFluidDataHigh.dCellEnd,
        mBoundaryData.Data->dPositions,
        mBoundaryData.dCellStart,
        mBoundaryData.dCellEnd,
        mFluidData.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
void Solver::computeAccelerationsAndUpdateStatesHigh ()
{
    computeAccelerationsAndUpdateStatesHighD<<<mFluidDataHigh.GridDimensions, 
        mFluidDataHigh.BlockDimensions>>>(
        mFluidDataHigh.dAccelerations,
        mFluidDataHigh.dTempStates,
        mFluidDataHigh.dDensities,
        mFluidDataHigh.dPressures,
        mFluidDataHigh.dTempPositions,
        mFluidDataHigh.dTempVelocities,
        mFluidData.dDensities,
        mFluidData.dPressures,
        mFluidData.dTempPositions,
        mFluidData.dTempVelocities,
        mFluidDataHigh.dCellStart,
        mFluidDataHigh.dCellEnd,
        mFluidData.dCellStart,
        mFluidData.dCellEnd,
        mBoundaryData.Data->dPositions,
        mBoundaryData.dCellStart,
        mBoundaryData.dCellEnd,
        mFluidDataHigh.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
void Solver::updateSystem (SPHParticleData& fluidData, float timeStep)
{
    // update:
    //  - active particles positions and velocities
    //  - active particles blendvalues according to their state
    //  - the IDs list (i.e. don't add (remove) particles, that have a 
    //    blend values of zero)


    // first set the number of particles to zero, as the following CUDA kernel
    // may delete particles from the system and counts the new amount of 
    // particles 
    CUDA::Memset<unsigned int>(fluidData.dNumParticles, 0, 1);

    updateSystemD<<<fluidData.GridDimensions, 
        fluidData.BlockDimensions>>>(
        fluidData.Data->dPositions,
        fluidData.dVelocities,
        fluidData.dBlendCoefficients,
        fluidData.dStates,
        fluidData.dActiveIDs,
        fluidData.dNumParticles,
        fluidData.dTempPositions,
        fluidData.dTempVelocities,
        fluidData.dTempBlendCoefficients,
        fluidData.dTempStates,
        fluidData.dAccelerations,
        timeStep,
        fluidData.Data->NumParticles
    );
}
//------------------------------------------------------------------------------
