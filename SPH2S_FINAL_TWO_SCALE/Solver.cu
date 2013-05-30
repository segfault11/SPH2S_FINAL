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
    LOW_RES = 0,
    HIGH_RES = 1
};
//------------------------------------------------------------------------------
#define COMPL(x) (x + 1) % 2
//------------------------------------------------------------------------------

//==============================================================================
// UTLITY device kernels definition
//==============================================================================

//------------------------------------------------------------------------------
__device__ void computeCoordinatesOff(
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
__device__ void computeCoordinates(
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
__device__ void computeHash(
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
__device__ void computeHash(
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
__device__ inline void computeNorm(float& norm, const float3& a)
{
    norm = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
//------------------------------------------------------------------------------
__device__ inline void computeDistance(
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
__device__ inline void evaluatePoly6Kernel(
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
__device__ inline void evaluateSpikyKernelGradient(
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
__device__ inline void evaluateViscosityKernelLaplacian(
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
__device__ inline void evaluateBoundaryWeight(
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
#define MEXICAN_HAT_C 0.8673250705840776f // c =  2/(sqrt(3)*pi^(1/4))
__device__ inline float evaluateMexicanHat3D(float x, float y, float z)
{
	x = x*x;
	y = y*y;
	z = z*z;
	return MEXICAN_HAT_C*(x+y+z-3.0f)*exp(-(x+y+z)/2.0f);
}
//------------------------------------------------------------------------------
__device__ inline void computeDensityCell(
    float& rhoi,                 // [out] density of particle i 
    const float3& xi,            // position of particle i
    const float* dPositions,     
    const float* dBlendCoefficients,
    unsigned int start,
    unsigned int end,
    int resID
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
        float lambdaj = dBlendCoefficients[j];
        float dist;
        computeDistance(dist, xi, xj);

        if (dist < gConfiguration.EffectiveRadius[resID])
        {
            float weight = 0.0f;
            evaluatePoly6Kernel(weight, dist, 
                gConfiguration.EffectiveRadius[resID]);
            rhoi += lambdaj*weight;
        }
            
    }
  
}
//------------------------------------------------------------------------------
__device__ inline void computeDensityCellCompl(
    float& rhoi,                 // [out] density of particle i 
    const float3& xi,            // position of particle i
    const float* dPositions,    
    const float* dBlendCoefficients, 
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
        float lambdaj = dBlendCoefficients[j];
        float dist;
        computeDistance(dist, xi, xj);

        if (dist < gConfiguration.EffectiveRadius[LOW_RES])
        {
            float weight0 = 0.0f;
            float weight1 = 0.0f;

            evaluatePoly6Kernel(weight0, dist, 
                gConfiguration.EffectiveRadius[LOW_RES]);
            evaluatePoly6Kernel(weight1, dist, 
                gConfiguration.EffectiveRadius[HIGH_RES]);
            rhoi += lambdaj*0.5f*(weight0 + weight1);
        }
            
    }
  
}
//------------------------------------------------------------------------------
__device__ inline void computeAccelerationCell(
    float3& fi,
    float3& velW,
    float& psiSum,
    float3& xc,
    float& massSum,
    float rhoi,
    float pi,
    const float3& xi,        
    const float3& vi,    
    const float* dDensities,
    const float* dPressures,
    const float* dPositions,     
    const float* dVelocities,
    const float* dBlendCoefficients,
    unsigned int start,
    unsigned int end,
    int resID
)
{
    for (unsigned int j = start; j < end; j++)
    {
        float3 xj;
        xj.x = dPositions[3*j + 0];
        xj.y = dPositions[3*j + 1];
        xj.z = dPositions[3*j + 2];
        float dist;
        float3 xij;
        xij.x = xi.x - xj.x; 
        xij.y = xi.y - xj.y; 
        xij.z = xi.z - xj.z; 
        computeNorm(dist, xij);
        
        if (dist != 0.0f && dist < gConfiguration.EffectiveRadius[resID])
        {
            float3 vj;
            vj.x = dVelocities[3*j + 0];
            vj.y = dVelocities[3*j + 1];
            vj.z = dVelocities[3*j + 2];
            float rhoj = dDensities[j];
            float pj = dPressures[j];
            float lambdaj = dBlendCoefficients[j];
            float mj = gConfiguration.FluidParticleMass[resID];

            // evaluate the pressure force partice j exerts on particle i
            float coeffP = -rhoi*mj*(pi/(rhoi*rhoi) + pj/(rhoj*rhoj));
            float3 grad;
            evaluateSpikyKernelGradient(
                grad, 
                xij,
                gConfiguration.EffectiveRadius[resID]
            );
            fi.x += lambdaj*coeffP*grad.x;
            fi.y += lambdaj*coeffP*grad.y;
            fi.z += lambdaj*coeffP*grad.z;

            // evaluate the viscosity force partice j exerts on particle i
            float coeffV = gConfiguration.Viscosity*mj/rhoj;
            float lapl = 0.0f;
            evaluateViscosityKernelLaplacian(
                lapl, 
                dist, 
                gConfiguration.EffectiveRadius[resID]
            );
            float3 vji;
            vji.x = vj.x - vi.x;
            vji.y = vj.y - vi.y;
            vji.z = vj.z - vi.z;
            fi.x += lambdaj*coeffV*vji.x*lapl;
            fi.y += lambdaj*coeffV*vji.y*lapl;
            fi.z += lambdaj*coeffV*vji.z*lapl;

            // evaluate the surface tension force partice j exerts on particle i
            float weight;
            evaluatePoly6Kernel(
                weight, 
                dist, 
                gConfiguration.EffectiveRadius[resID]
            );
            float coeffT = -weight*mj*gConfiguration.TensionCoefficient;
        
            fi.x += lambdaj*coeffT*xij.x;
            fi.y += lambdaj*coeffT*xij.y;
            fi.z += lambdaj*coeffT*xij.z;

            float h = gConfiguration.EffectiveRadius[resID];
            float psi = evaluateMexicanHat3D(xij.x/h, xij.y/h, xij.z/h);
            velW.x += vj.x*psi;
            velW.y += vj.y*psi;
            velW.z += vj.z*psi;
            psiSum += psi;

            xc.x += mj*xj.x;
            xc.y += mj*xj.y;
            xc.z += mj*xj.z;
            massSum += mj;
        }

    }

}
//------------------------------------------------------------------------------
__device__ inline void computeAccelerationCellCompl(
    float3& fi,
    float rhoi,
    float pi,
    const float3& xi,        
    const float3& vi,    
    const float* dDensities,
    const float* dPressures,
    const float* dPositions,     
    const float* dVelocities,
    const float* dBlendCoefficients,
    unsigned int start,
    unsigned int end,
    int resID         // resolution id for complement domain
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
        float lambdaj = dBlendCoefficients[j];
        float dist;
        float3 xij;
        xij.x = xi.x - xj.x; 
        xij.y = xi.y - xj.y; 
        xij.z = xi.z - xj.z; 
        computeNorm(dist, xij);
        
        if (dist != 0.0f && dist < gConfiguration.EffectiveRadius[LOW_RES])
        {
            // evaluate the pressure force partice j exerts on particle i
            float coeffP = -rhoi*gConfiguration.FluidParticleMass[resID]*
                (pi/(rhoi*rhoi) + pj/(rhoj*rhoj));
            float3 grad0;
            float3 grad1;
            evaluateSpikyKernelGradient(
                grad0, 
                xij,
                gConfiguration.EffectiveRadius[LOW_RES]
            );
            evaluateSpikyKernelGradient(
                grad1, 
                xij,
                gConfiguration.EffectiveRadius[HIGH_RES]
            );
            fi.x += lambdaj*coeffP*0.5*(grad0.x + grad1.x);
            fi.y += lambdaj*coeffP*0.5*(grad0.y + grad1.y);
            fi.z += lambdaj*coeffP*0.5*(grad0.z + grad1.z);

            // evaluate the viscosity force partice j exerts on particle i
            float coeffV = gConfiguration.Viscosity*
                gConfiguration.FluidParticleMass[resID]/rhoj;
            float lapl0 = 0.0f;
            float lapl1 = 0.0f;
            evaluateViscosityKernelLaplacian(
                lapl0, 
                dist, 
                gConfiguration.EffectiveRadius[LOW_RES]
            );
            evaluateViscosityKernelLaplacian(
                lapl1, 
                dist, 
                gConfiguration.EffectiveRadius[HIGH_RES]
            );
            float3 vji;
            vji.x = vj.x - vi.x;
            vji.y = vj.y - vi.y;
            vji.z = vj.z - vi.z;
            float lapl = 0.5f*(lapl0 + lapl1);
            fi.x += lambdaj*coeffV*vji.x*lapl;
            fi.y += lambdaj*coeffV*vji.y*lapl;
            fi.z += lambdaj*coeffV*vji.z*lapl;

            // evaluate the surface tension force partice j exerts on particle i
            float weight0;
            float weight1;
            evaluatePoly6Kernel(
                weight0, 
                dist, 
                gConfiguration.EffectiveRadius[LOW_RES]
            );
            evaluatePoly6Kernel(
                weight1, 
                dist, 
                gConfiguration.EffectiveRadius[HIGH_RES]
            );
            float weight = 0.5f*(weight0 + weight1);
            float coeffT = -weight*gConfiguration.FluidParticleMass[resID]*
                gConfiguration.TensionCoefficient;
        
            fi.x += lambdaj*coeffT*xij.x;
            fi.y += lambdaj*coeffT*xij.y;
            fi.z += lambdaj*coeffT*xij.z;
        }

    }

}
//------------------------------------------------------------------------------
__device__ void computeBoundaryForceCell(
    float3& bi,
    const float3& xi,
    const float* dPositions,     
    unsigned int start,
    unsigned int end,
    int resID
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

        if (dist < gConfiguration.EffectiveRadius[resID])
        {
            float weight = 0.0f;
            evaluateBoundaryWeight(
                weight, 
                dist, 
                gConfiguration.EffectiveRadius[resID]
            );
            weight*= gConfiguration.BoundaryParticleMass/
                (gConfiguration.FluidParticleMass[resID] + 
                gConfiguration.BoundaryParticleMass);
            bi.x += weight*xij.x/dist;
            bi.y += weight*xij.y/dist;
            bi.z += weight*xij.z/dist;
        }
    }
}
//------------------------------------------------------------------------------
#define SQRT3INV 0.57735026919
__device__ void insertHighResParticles(
    float* dPositions,              // positions of the high res particles
    float* dVelocities,             // velocities of the high res particles
    float* dBlendCoefficients,      // blend vals of the high res particles
    int* dStates,         // states of the high res particles
    unsigned int* dActiveIDs,       // list of active high particle ids
    unsigned int* dNumParticles,    // # particles in the list
    const float3& posLow,           // position of the particle that is split
    const float3& velLow,           // velocity of the particle that is split
    float densLow                   // density of the particle that is split
)
{
    const float dir[] = {
         SQRT3INV,  SQRT3INV,  SQRT3INV,
         SQRT3INV,  SQRT3INV, -SQRT3INV,
         SQRT3INV, -SQRT3INV,  SQRT3INV,
         SQRT3INV, -SQRT3INV, -SQRT3INV,
        -SQRT3INV,  SQRT3INV,  SQRT3INV,
        -SQRT3INV,  SQRT3INV, -SQRT3INV,
        -SQRT3INV, -SQRT3INV,  SQRT3INV,
        -SQRT3INV, -SQRT3INV, -SQRT3INV
    };
    
    // add eight to the high res particles
    unsigned int id = atomicAdd(dNumParticles, 8);
    float r = 0.5f*pow(
            3.0f/(4.0f*float(M_PI))*
            gConfiguration.FluidParticleMass[LOW_RES]/densLow,
            1.0f/3.0f
        );

    for (unsigned int i = 0; i < 8; i++)
    {
        dActiveIDs[id + i] = id + i;
        dBlendCoefficients[id + i] = 0.0f;
        dStates[id + i] = 0x06;
        dPositions[3*(id + i) + 0] = posLow.x + r*dir[3*i + 0];
        dPositions[3*(id + i) + 1] = posLow.y + r*dir[3*i + 1];
        dPositions[3*(id + i) + 2] = posLow.z + r*dir[3*i + 2];
        dVelocities[3*(id + i) + 0] = velLow.x;
        dVelocities[3*(id + i) + 1] = velLow.y;
        dVelocities[3*(id + i) + 2] = velLow.z;
    }

}
//------------------------------------------------------------------------------

//==============================================================================
// GLOBAL device kernel definitions
//==============================================================================

//------------------------------------------------------------------------------
__global__ void computeHashs(
    unsigned int* dHashs,           // hash values of each particle
    unsigned int* dActiveIDs,       // array of active particle ids
    const float* dPositions,        // positions of each particle 
    unsigned int numParticles,      // number of ids in the id array
    int resID
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

    computeHash(dHashs[idx], pos, gConfiguration.Grid[resID]);
};
//------------------------------------------------------------------------------
__global__ void reorderComputeCellStartEndBoundaryD(
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
__global__ void reorderAndComputeCellStartEndD(
    unsigned int* dCellStart,
    unsigned int* dCellEnd,
    float* dTempPositions,
    float* dTempVelocities,
    int* dTempStates,
    float* dTempBlendCoefficients,
    const unsigned int* dSortedIDs,
    const float* dPositions,
    const float* dVelocities,
    const int* dStates,
    const float* dBlendCoefficients,
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
    dTempStates[idx] = dStates[id];
    dTempBlendCoefficients[idx] = dBlendCoefficients[id];
    
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
__global__ void computeDensitiesPressuresD(
    float* dDensities,              // [out] computed densities
    float* dPressures,
    const float* dPositions,
    const float* dBlendCoefficients,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const float* dPositionsCompl,
    const float* dBlendCoefficientsCompl,
    const unsigned int* dCellStartCompl,
    const unsigned int* dCellEndCompl,
    unsigned int numParticles,
    int resID             // resolution identifier
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
    float rhoi = 0.0f;
    float rhoiCompl = 0.0f;
    int3 cs, ce, cc;

    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[resID], 
        -gConfiguration.EffectiveRadius[resID]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[resID], 
        gConfiguration.EffectiveRadius[resID]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[resID]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeDensityCell(
                    rhoi,
                    xi,
                    dPositions,
                    dBlendCoefficients,
                    start,
                    end,
                    resID
                );
            }
        }
    }

    //--------------------------------------------------------------------------
    // compute density contribution of the complementary domain
    //--------------------------------------------------------------------------

    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[COMPL(resID)], 
        -gConfiguration.EffectiveRadius[LOW_RES]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[COMPL(resID)], 
        gConfiguration.EffectiveRadius[LOW_RES]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[COMPL(resID)]);
                unsigned int start = dCellStartCompl[hash];
                unsigned int end = dCellEndCompl[hash];

                computeDensityCellCompl(
                    rhoiCompl,
                    xi,
                    dPositionsCompl,
                    dBlendCoefficientsCompl,
                    start,
                    end
                );
            }
        }
    }


    rhoi = rhoi*gConfiguration.FluidParticleMass[resID] + 
        rhoiCompl*gConfiguration.FluidParticleMass[COMPL(resID)];
    dDensities[idx] = rhoi;
    dPressures[idx] = gConfiguration.BulkModulus*
        (rhoi - gConfiguration.RestDensity);
}
//------------------------------------------------------------------------------
__global__ void computeAccelerationsD(
    float* dAccelerations,
    float* dColorValues,
    const float* dDensities,              
    const float* dPressures,
    const float* dPositions,
    const float* dVelocities,
    const float* dBlendCoefficients,
    int* dStates,
    const unsigned int* dCellStart,
    const unsigned int* dCellEnd,
    const float* dDensitiesCompl,              
    const float* dPressuresCompl,
    const float* dPositionsCompl,
    const float* dVelocitiesCompl,
    const float* dBlendCoefficientsCompl,
    const unsigned int* dCellStartCompl,
    const unsigned int* dCellEndCompl,
    const float* dBoundaryPositions,
    const unsigned int* dBoundaryCellStart,
    const unsigned int* dBoundaryCellEnd,
    unsigned int numParticles,
    int resID
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
    float psi = evaluateMexicanHat3D(0.0f, 0.0f, 0.0f);
    float psiSum = psi;
    float3 velW;
    velW.x = vi.x*psi;
    velW.y = vi.y*psi;
    velW.z = vi.z*psi;
    float massSum = gConfiguration.FluidParticleMass[resID];
    float3 xc;
    xc.x = gConfiguration.FluidParticleMass[resID]*xi.x;
    xc.y = gConfiguration.FluidParticleMass[resID]*xi.y; 
    xc.z = gConfiguration.FluidParticleMass[resID]*xi.z;
    int3 cc, cs, ce;

    //--------------------------------------------------------------------------
    // compute force contribution of the same domain
    //--------------------------------------------------------------------------
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[resID], 
        -gConfiguration.EffectiveRadius[resID]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[resID], 
        gConfiguration.EffectiveRadius[resID]
    );
    
    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[resID]);
                unsigned int start = dCellStart[hash];
                unsigned int end = dCellEnd[hash];

                computeAccelerationCell(
                    fi,
                    velW,
                    psiSum,
                    xc,
                    massSum,
                    rhoi,
                    pi,
                    xi,
                    vi,
                    dDensities,
                    dPressures,
                    dPositions,
                    dVelocities,
                    dBlendCoefficients,
                    start,
                    end,
                    resID
                );
            }
        }
    }

    //--------------------------------------------------------------------------
    // compute force contribution of the complementary domain
    //--------------------------------------------------------------------------
    
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[COMPL(resID)], 
        -gConfiguration.EffectiveRadius[LOW_RES]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[COMPL(resID)], 
        gConfiguration.EffectiveRadius[LOW_RES]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[COMPL(resID)]);
                unsigned int start = dCellStartCompl[hash];
                unsigned int end = dCellEndCompl[hash];

                computeAccelerationCellCompl(
                    fi,
                    rhoi,
                    pi,
                    xi,
                    vi,
                    dDensitiesCompl,
                    dPressuresCompl,
                    dPositionsCompl,
                    dVelocitiesCompl,
                    dBlendCoefficientsCompl,
                    start,
                    end,
                    COMPL(resID)
                );
            }
        }
    }

    //--------------------------------------------------------------------------
    // compute force contribution of the boundary
    //--------------------------------------------------------------------------
    
    computeCoordinatesOff(
        cs, 
        xi, 
        gConfiguration.Grid[LOW_RES], 
        -gConfiguration.EffectiveRadius[LOW_RES]
    );
    computeCoordinatesOff(
        ce, 
        xi, 
        gConfiguration.Grid[LOW_RES], 
        gConfiguration.EffectiveRadius[LOW_RES]
    );

    for (cc.z = cs.z; cc.z <= ce.z; cc.z++)
    {
        for (cc.y = cs.y; cc.y <= ce.y; cc.y++)
        {
            for (cc.x  = cs.x; cc.x <= ce.x; cc.x++)
            {
                unsigned int hash;
                computeHash(hash, cc, gConfiguration.Grid[LOW_RES]);
                unsigned int start = dBoundaryCellStart[hash];
                unsigned int end = dBoundaryCellEnd[hash];
                computeBoundaryForceCell(
                    bi,
                    xi,
                    dBoundaryPositions,
                    start,
                    end,
                    LOW_RES
                );
            }
        }
    }

    dAccelerations[3*idx + 0] = fi.x/rhoi + bi.x;
    dAccelerations[3*idx + 1] = fi.y/rhoi - 9.81f + bi.y;
    dAccelerations[3*idx + 2] = fi.z/rhoi + bi.z;

    xc.x /= massSum;
    xc.y /= massSum;
    xc.z /= massSum;

    float3 xd;
    xd.x = xi.x - xc.x;
    xd.y = xi.y - xc.y;
    xd.z = xi.z - xc.z;
    float disticm;

    computeNorm(disticm, xd);

    float ene = 1/(psiSum*psiSum*gConfiguration.EffectiveRadius[resID])*
        (velW.x*velW.x + velW.y*velW.y + velW.z*velW.z);
    
    float maxEne = 300.0f;
    ene = min(ene, maxEne);
    
    if (disticm < 0.0025f)
    {
        dColorValues[idx] = 0.3f;
    }
    else
    {
        dColorValues[idx] = ene/maxEne;    
    }

    int state = dStates[idx];

    if (state == 8 && resID == 0)
    {
    //    dStates[idx] = state | 3;   // mark particle for splitting
    }
}
//------------------------------------------------------------------------------
__global__ void integrateD(
    float* dPositions, 
    float* dVelocities, 
    float* dAccelerations,
    int* dStates,
    float* dBlendCoefficients,
    unsigned int* dActiveIDs,               // array of active particle ids
    unsigned int* dNumParticles,            // counter var for active particles
    float* dPositionsCompl, 
    float* dVelocitiesCompl, 
    int* dStatesCompl,
    float* dBlendCoefficientsCompl,
    unsigned int* dActiveIDsCompl,         
    unsigned int* dNumParticlesCompl,    
    const float* dTempPositions,
    const float* dTempVelocities,
    const float* dDensities,
    const int* dTempStates,
    const float* dTempBlendCoefficients,
    float timeStep,
    unsigned int numParticles,
    int resID
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    //--------------------------------------------------------------------------
    // update posititions and velocities
    //--------------------------------------------------------------------------

    float3 xi;
    xi.x = dTempPositions[3*idx + 0];
    xi.y = dTempPositions[3*idx + 1];
    xi.z = dTempPositions[3*idx + 2];
    float3 vi;
    vi.x = dTempVelocities[3*idx + 0];
    vi.y = dTempVelocities[3*idx + 1];
    vi.z = dTempVelocities[3*idx + 2];

    vi.x += timeStep*dAccelerations[3*idx + 0]; 
    vi.y += timeStep*dAccelerations[3*idx + 1]; 
    vi.z += timeStep*dAccelerations[3*idx + 2]; 


    // addjust the length of the velocity vector to avoid 
    // small instabilities
    float norm;
    computeNorm(norm, vi);
    float vscale = min(
            norm, 
            0.8f*gConfiguration.EffectiveRadius[resID]/timeStep
        );
    
    vi.x *= vscale/norm;
    vi.y *= vscale/norm;
    vi.z *= vscale/norm;

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

    //--------------------------------------------------------------------------
    // update blend coefficients and states
    //--------------------------------------------------------------------------

    const float c[] = {0.0f, -1.0f, 1.0f, 0.0f}; 

    int state = dTempStates[idx];
    float blendCoeff = dTempBlendCoefficients[idx];
    
    blendCoeff += c[state & 0x03]*gConfiguration.BlendIncrement;

    if (blendCoeff <= 0.0f && (state & 0x03) == 1)
    {
        dBlendCoefficients[idx] = blendCoeff;
        dStates[idx] = state;
        return;
    }

    if (blendCoeff >= 1.0f && (state & 0x03) == 2)
    {
        // if the particle has/reaches a blend coeff of 1.0f or above
        // it stays/becomes a default particle (0)
        state = state & 0x04; 
        blendCoeff = 1.0f;
    }

    // low res particles which are marked for splitting are splitted here
    if ((state & 0x03) == 3 && resID == LOW_RES)
    {
        float dens = dDensities[idx];

        insertHighResParticles(
            dPositionsCompl,
            dVelocitiesCompl,
            dBlendCoefficientsCompl,
            dStatesCompl,
            dActiveIDsCompl,
            dNumParticlesCompl,
            xi,
            vi,
            dens
        );

        state = 0x01; // set state to delete (1) and 
    }

    dBlendCoefficients[idx] = blendCoeff;
    dStates[idx] = state;

    unsigned int i = atomicAdd(dNumParticles, 1);
    dActiveIDs[i] = idx;
}
//------------------------------------------------------------------------------

//==============================================================================
//  HOST code starts here 
//==============================================================================

//------------------------------------------------------------------------------
#define BLOCK_DIMENSIONS_X 256
#define EMPTY_CELL_ID 0xFF
//------------------------------------------------------------------------------

//==============================================================================
//  UTILITY functions start here
//==============================================================================

//------------------------------------------------------------------------------
void computeGridDimensions(
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
    CUDA::Fill<float>(dDensities, data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dPressures, data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dAccelerations, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dVelocities, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dTempPositions, 3*data->MaxParticles, 0.0f);
    CUDA::Fill<float>(dTempVelocities, 3*data->MaxParticles, 0.0f);
    CUDA::Alloc<unsigned int>(&dActiveIDs, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dHashs, data->MaxParticles);
    CUDA::Alloc<unsigned int>(&dCellStart, numGridCells);
    CUDA::Alloc<unsigned int>(&dCellEnd, numGridCells);
    CUDA::Fill<unsigned int>(dActiveIDs, data->MaxParticles, 0, 1);
    CUDA::Fill<unsigned int>(dHashs, data->MaxParticles, 0);
    CUDA::Fill<unsigned int>(dCellStart, numGridCells, 0);
    CUDA::Fill<unsigned int>(dCellEnd, numGridCells, 0);
    CUDA::Alloc<int>(&dTempStates, data->MaxParticles);
    CUDA::Alloc<float>(&dBlendCoefficients, data->MaxParticles);
    CUDA::Alloc<float>(&dTempBlendCoefficients, data->MaxParticles);
    CUDA::Fill<int>(dTempStates, data->MaxParticles, 0);
    CUDA::Fill<float>(dBlendCoefficients, data->MaxParticles, 1.0f);
    CUDA::Fill<float>(dTempBlendCoefficients, data->MaxParticles, 1.0f);
    CUDA::Alloc<unsigned int>(&dNumParticles, 1);
    CUDA::Memcpy<unsigned int>(
        dNumParticles, 
        &Data->NumParticles, 
        1, 
        cudaMemcpyHostToDevice
    );

    // compute the number of cuda blocks we need based on the current number
    // of particles and the threads per block we use. Also compute the amount
    // of shared memory we need to compute the values for [dCellStart] and
    // [dCellEnd]
    computeGridDimensions(GridDimensions, BlockDimensions, data->MaxParticles);
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

    CUDA::Free<unsigned int>(&dHashs);
    CUDA::Free<unsigned int>(&dCellStart);
    CUDA::Free<unsigned int>(&dCellEnd);

    CUDA::Free<int>(&dTempStates);
    CUDA::Free<float>(&dBlendCoefficients);
    CUDA::Free<float>(&dTempBlendCoefficients);

    CUDA::Free<unsigned int>(&dNumParticles);
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
Solver::Solver(
    ParticleData* fluidData, 
    ParticleData* fluidDataHigh,
    ParticleData* boundaryData,
    const SolverConfiguration* configuration
)
: 
    mConfiguration(*configuration)
{
    // store pointer to fluid particles and boundary particles also store
    // a copy of solver configuration

    mFluidData[0] = new SPHParticleData(
            fluidData, 
            Grid::ComputeNumGridCells(configuration->Grid[0])
        );
    mFluidData[1] = new SPHParticleData(
            fluidDataHigh, 
            Grid::ComputeNumGridCells(configuration->Grid[1])
        );
    mBoundaryData = new BoundaryParticleData(
            boundaryData, 
            Grid::ComputeNumGridCells(configuration->Grid[0])
        );

    // compute neighborhood of boundary particles beforehand 
    unsigned int* dBoundaryIDs;
    CUDA::Alloc<unsigned int>(&dBoundaryIDs, mBoundaryData->Data->MaxParticles);
    CUDA::Fill<unsigned int>(dBoundaryIDs, mBoundaryData->Data->MaxParticles, 
        0, 1);

    float* dBoundaryPositions;
    CUDA::Alloc<float>(
        &dBoundaryPositions, 
        3*mBoundaryData->Data->MaxParticles
    );
    
    this->Bind();   // bind first
    mBoundaryData->Data->Map();
    computeHashs<<<mBoundaryData->GridDimensions, 
        mBoundaryData->BlockDimensions>>>(
        mBoundaryData->dHashs, 
        dBoundaryIDs, 
        mBoundaryData->Data->dPositions,
        mBoundaryData->Data->NumParticles,
        0
    );
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(mBoundaryData->dHashs),
        thrust::device_ptr<unsigned int>(mBoundaryData->dHashs + 
            mBoundaryData->Data->NumParticles),
        thrust::device_ptr<unsigned int>(dBoundaryIDs)
    );
    CUDA::Memset<unsigned int>(
        mBoundaryData->dCellStart, 
        EMPTY_CELL_ID, 
        mBoundaryData->NumGridCells
    );
    CUDA::Memset<unsigned int>(
        mBoundaryData->dCellEnd, 
        EMPTY_CELL_ID, 
        mBoundaryData->NumGridCells
    );
    reorderComputeCellStartEndBoundaryD<<<mBoundaryData->GridDimensions,
        mBoundaryData->BlockDimensions, 
        mBoundaryData->SharedMemSize>>>(
        mBoundaryData->dCellStart,
        mBoundaryData->dCellEnd,
        dBoundaryPositions,
        mBoundaryData->Data->dPositions,
        dBoundaryIDs,
        mBoundaryData->dHashs,
        mBoundaryData->Data->NumParticles 
    );
    CUDA::Memcpy<float>(
        mBoundaryData->Data->dPositions, 
        dBoundaryPositions, 
        3*mBoundaryData->Data->MaxParticles,
        cudaMemcpyDeviceToDevice
    );
    mBoundaryData->Data->Unmap();

    CUDA::Free<unsigned int>(&dBoundaryIDs);
    CUDA::Free<float>(&dBoundaryPositions);

    // set particle states to either low or high
    mFluidData[0]->Data->Map();
    CUDA::Fill<int>(
        mFluidData[0]->Data->dStates, 
        mFluidData[0]->Data->MaxParticles,
        0
    );
    mFluidData[0]->Data->Unmap();
    mFluidData[1]->Data->Map();
    CUDA::Fill<int>(
        mFluidData[1]->Data->dStates, 
        mFluidData[1]->Data->MaxParticles,
        4
    );
    mFluidData[1]->Data->Unmap();
}
//------------------------------------------------------------------------------
Solver::~Solver()
{
    delete mFluidData[LOW_RES];
    delete mFluidData[HIGH_RES];
    delete mBoundaryData;
}
//------------------------------------------------------------------------------
void Solver::Bind() const
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
void Solver::Advance(float timeStep)
{
    //std::cout << mFluidData[LOW_RES]->Data->NumParticles << " " 
    //    << mFluidData[HIGH_RES]->Data->NumParticles << std::endl;

    CUDA::Timer t;
    t.Start();
    mFluidData[LOW_RES]->Data->Map();
    mFluidData[HIGH_RES]->Data->Map();
    mBoundaryData->Data->Map();
    this->computeNeighborhoods(HIGH_RES);
    this->computeNeighborhoods(LOW_RES);
    this->computeDensities(HIGH_RES);
    this->computeDensities(LOW_RES);
    this->computeAccelerations(HIGH_RES);
    this->computeAccelerations(LOW_RES);
    //CUDA::DumpArray<float>(mFluidData[LOW_RES]->Data->dColorValues, mFluidData[LOW_RES]->Data->NumParticles);
    CUDA::Memset<unsigned int>(mFluidData[HIGH_RES]->dNumParticles, 0, 1);
    CUDA::Memset<unsigned int>(mFluidData[LOW_RES]->dNumParticles, 0, 1);
    this->integrate(HIGH_RES, timeStep);
    this->integrate(LOW_RES, timeStep);
    mBoundaryData->Data->Unmap();
    mFluidData[HIGH_RES]->Data->Unmap();
    mFluidData[LOW_RES]->Data->Unmap();
    t.Stop();
    t.DumpElapsed();
    //std::system("pause");
}
//------------------------------------------------------------------------------
void Solver::computeNeighborhoods(int resID)
{
    // reset, then compute cell start end list
    // NOTE: it is important to reset the cell lists, even though zero particles
    // are active, as otherwise the lists would still contain the particle ids
    // of the last particles before complete deletion of the system
    CUDA::Memset<unsigned int>(
        mFluidData[resID]->dCellStart, 
        EMPTY_CELL_ID, 
        mFluidData[resID]->NumGridCells
    );
    CUDA::Memset<unsigned int>(
        mFluidData[resID]->dCellEnd, 
        EMPTY_CELL_ID, 
        mFluidData[resID]->NumGridCells
    );

    CUDA::Memcpy<unsigned int>(
        &mFluidData[resID]->Data->NumParticles,
        mFluidData[resID]->dNumParticles,
        1,
        cudaMemcpyDeviceToHost
    );

    if (!mFluidData[resID]->Data->NumParticles)
    {
        return;
    }

    // compute hashs of all particles
    computeHashs<<<mFluidData[resID]->GridDimensions,
        mFluidData[resID]->BlockDimensions>>>(
        mFluidData[resID]->dHashs,
        mFluidData[resID]->dActiveIDs,
        mFluidData[resID]->Data->dPositions,
        mFluidData[resID]->Data->NumParticles,
        resID
    );
        
    // sort the active particle ids by their hash
    thrust::sort_by_key(
        thrust::device_ptr<unsigned int>(mFluidData[resID]->dHashs),
        thrust::device_ptr<unsigned int>(mFluidData[resID]->dHashs + 
            mFluidData[resID]->Data->NumParticles),
        thrust::device_ptr<unsigned int>(mFluidData[resID]->dActiveIDs)
    );

    reorderAndComputeCellStartEndD<<<mFluidData[resID]->GridDimensions, 
        mFluidData[resID]->BlockDimensions, mFluidData[resID]->SharedMemSize>>>(
        mFluidData[resID]->dCellStart,
        mFluidData[resID]->dCellEnd,
        mFluidData[resID]->dTempPositions,
        mFluidData[resID]->dTempVelocities,
        mFluidData[resID]->dTempStates,
        mFluidData[resID]->dTempBlendCoefficients,
        mFluidData[resID]->dActiveIDs,
        mFluidData[resID]->Data->dPositions,
        mFluidData[resID]->dVelocities,
        mFluidData[resID]->Data->dStates,
        mFluidData[resID]->dBlendCoefficients,
        mFluidData[resID]->dHashs,
        mFluidData[resID]->Data->NumParticles
    );

}
//------------------------------------------------------------------------------
void Solver::computeDensities (int resID)
{
    if (!mFluidData[resID]->Data->NumParticles)
    {
        return;
    }

    computeDensitiesPressuresD<<<mFluidData[resID]->GridDimensions, 
        mFluidData[resID]->BlockDimensions>>>(
        mFluidData[resID]->dDensities,
        mFluidData[resID]->dPressures,
        mFluidData[resID]->dTempPositions,
        mFluidData[resID]->dTempBlendCoefficients,
        mFluidData[resID]->dCellStart,
        mFluidData[resID]->dCellEnd,
        mFluidData[COMPL(resID)]->dTempPositions,
        mFluidData[COMPL(resID)]->dTempBlendCoefficients,
        mFluidData[COMPL(resID)]->dCellStart,
        mFluidData[COMPL(resID)]->dCellEnd,
        mFluidData[resID]->Data->NumParticles,
        resID
    );
}
//------------------------------------------------------------------------------
void Solver::computeAccelerations(int resID)
{
    if (!mFluidData[resID]->Data->NumParticles)
    {
        return;
    }

    computeAccelerationsD<<<mFluidData[resID]->GridDimensions, 
        mFluidData[resID]->BlockDimensions>>>(
        mFluidData[resID]->dAccelerations,
        mFluidData[resID]->Data->dColorValues,
        mFluidData[resID]->dDensities,
        mFluidData[resID]->dPressures,
        mFluidData[resID]->dTempPositions,
        mFluidData[resID]->dTempVelocities,
        mFluidData[resID]->dTempBlendCoefficients,
        mFluidData[resID]->dTempStates,
        mFluidData[resID]->dCellStart,
        mFluidData[resID]->dCellEnd,
        mFluidData[COMPL(resID)]->dDensities,
        mFluidData[COMPL(resID)]->dPressures,
        mFluidData[COMPL(resID)]->dTempPositions,
        mFluidData[COMPL(resID)]->dTempVelocities,
        mFluidData[COMPL(resID)]->dTempBlendCoefficients,
        mFluidData[COMPL(resID)]->dCellStart,
        mFluidData[COMPL(resID)]->dCellEnd,
        mBoundaryData->Data->dPositions,
        mBoundaryData->dCellStart,
        mBoundaryData->dCellEnd,
        mFluidData[resID]->Data->NumParticles,
        resID
    );
}
//------------------------------------------------------------------------------
void Solver::integrate(int resID, float timeStep)
{
    if (!mFluidData[resID]->Data->NumParticles)
    {
        return;
    }

    integrateD<<<mFluidData[resID]->GridDimensions, 
        mFluidData[resID]->BlockDimensions>>>(
        mFluidData[resID]->Data->dPositions,
        mFluidData[resID]->dVelocities,
        mFluidData[resID]->dAccelerations,
        mFluidData[resID]->Data->dStates,
        mFluidData[resID]->dBlendCoefficients,
        mFluidData[resID]->dActiveIDs,
        mFluidData[resID]->dNumParticles,
        mFluidData[COMPL(resID)]->Data->dPositions,
        mFluidData[COMPL(resID)]->dVelocities,
        mFluidData[COMPL(resID)]->Data->dStates,
        mFluidData[COMPL(resID)]->dBlendCoefficients,
        mFluidData[COMPL(resID)]->dActiveIDs,
        mFluidData[COMPL(resID)]->dNumParticles,
        mFluidData[resID]->dTempPositions,
        mFluidData[resID]->dTempVelocities,
        mFluidData[resID]->dDensities,
        mFluidData[resID]->dTempStates,
        mFluidData[resID]->dTempBlendCoefficients,
        timeStep,
        mFluidData[resID]->Data->NumParticles,
        resID
    );
}
//------------------------------------------------------------------------------
