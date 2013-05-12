//------------------------------------------------------------------------------
//  Solver.h
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
#include <cmath>
#include "ParticleData.h"
#include "cuda.h"
#include "Grid.h"
//------------------------------------------------------------------------------
#define M_PI 3.14159265358979323846
//------------------------------------------------------------------------------
struct SolverConfiguration
{
    static SolverConfiguration MakeConfiguration (
        float3 origin,                  // domain origin
        float3 end,                     // domain end
        float volume,                   // fluid volume
        unsigned int numParticles, 
        unsigned int avgParticles,
        float restDensity,
        float bulkModulus,
        float viscosity,
        float speedSound,
        float tensionCoefficient,
        float epsilon
    )
    {
        SolverConfiguration config;

        config.EffectiveRadius = std::powf(
                0.75f*volume*static_cast<float>(avgParticles)/
                (M_PI*static_cast<float>(numParticles)), 
                1.0f/3.0f
            ); 
        config.FluidParticleMass = restDensity*volume/
            static_cast<float>(numParticles);

        config.BoundaryParticleMass = config.FluidParticleMass;
        config.RestDensity = restDensity;
        config.Grid = Grid::MakeGrid(origin, end, config.EffectiveRadius);
        config.BulkModulus = bulkModulus;
        config.Viscosity = viscosity;
        config.SpeedSound = speedSound;
        config.TensionCoefficient = tensionCoefficient;
        config.XSPHCoeff = epsilon;
        return config;
    };

    float EffectiveRadius;
    float FluidParticleMass;
    float BoundaryParticleMass;
    float RestDensity;
    float BulkModulus;
    float Viscosity;
    float SpeedSound;
    float TensionCoefficient;
    float XSPHCoeff;
    ::Grid Grid;                // simulation grid
};
//------------------------------------------------------------------------------
class Solver 
{
    class SPHParticleData
    {
    // stores particle data and additional information needed for the simulation
    public:
        SPHParticleData (ParticleData* data, unsigned int numGridCells);
        ~SPHParticleData ();

        ::ParticleData* Data;

        float* dDensities;
        float* dPressures;
        float* dAccelerations;
        float* dVelocities; 

        float* dTempPositions;      // temporary positions used for actual computations
        float* dTempVelocities;     // temporary velocities used for actual computations

        unsigned int* dActiveIDs;
        unsigned int* dHashs;
        unsigned int* dCellStart;
        unsigned int* dCellEnd;
        unsigned int NumGridCells;

        // cuda kernel information
        dim3 BlockDimensions;
        dim3 GridDimensions;
        int SharedMemSize;
    };

    class BoundaryParticleData
    {
    public:
        BoundaryParticleData (ParticleData* data, unsigned int numGridCells);
        ~BoundaryParticleData ();
        
        ::ParticleData* Data;
            
        unsigned int* dHashs;
        unsigned int* dCellStart;
        unsigned int* dCellEnd;
        unsigned int NumGridCells;
        
        // cuda kernel information
        dim3 BlockDimensions;
        dim3 GridDimensions;
        int SharedMemSize;    
    };

public:
    Solver(
        ParticleData* fluidData, 
        ParticleData* boundaryData,
        const SolverConfiguration* configuration
    );
    ~Solver();

    void Bind() const;
    void Advance (float dt);

private:
    inline void computeNeighborhoods ();
    inline void computeDensities ();
    inline void computeAccelerations ();
    inline void integrate (float timeStep);

    SolverConfiguration mConfiguration;
    SPHParticleData mFluidData;
    BoundaryParticleData mBoundaryData;
};
//------------------------------------------------------------------------------