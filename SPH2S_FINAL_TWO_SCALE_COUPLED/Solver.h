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
    static SolverConfiguration MakeConfiguration(
        float3 origin,                  // domain origin
        float3 end,                     // domain end
        float volume,                   // fluid volume
        unsigned int numParticles, 
        unsigned int avgParticles,
        float restDensity,
        float bulkModulus,
        float viscosity,
        float speedSound,
        float tensionCoefficient
    )
    {
        SolverConfiguration config;
        config.EffectiveRadius[0] = std::powf(
                0.75f*volume*static_cast<float>(avgParticles)/
                (M_PI*static_cast<float>(numParticles)), 
                1.0f/3.0f
            ); 
        config.EffectiveRadius[1] = 0.5f*config.EffectiveRadius[0];
        config.FluidParticleMass[0] = restDensity*volume/
            static_cast<float>(numParticles);
        config.FluidParticleMass[1] = 1.0f/8.0f*config.FluidParticleMass[0];
        config.BoundaryParticleMass = config.FluidParticleMass[0];
        config.RestDensity = restDensity;
        config.Grid[0] = Grid::MakeGrid(origin, end, config.EffectiveRadius[0]);
        config.Grid[1] = Grid::MakeGrid(origin, end, config.EffectiveRadius[1]);
        config.BulkModulus = bulkModulus;
        config.Viscosity = viscosity;
        config.SpeedSound = speedSound;
        config.TensionCoefficient = tensionCoefficient;
        return config;
    };

    float EffectiveRadius[2];
    float FluidParticleMass[2];
    float BoundaryParticleMass;
    float RestDensity;
    float BulkModulus;
    float Viscosity;
    float SpeedSound;
    float TensionCoefficient;
    ::Grid Grid[2];                // simulation grid
};
//------------------------------------------------------------------------------
class Solver 
{
    class SPHParticleData
    {
    // stores particle data and additional information needed for the simulation
    public:
        SPHParticleData(ParticleData* data, unsigned int numGridCells);
        ~SPHParticleData();

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
        BoundaryParticleData(ParticleData* data, unsigned int numGridCells);
        ~BoundaryParticleData();
        
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
        ParticleData* fluidDataHigh,
        ParticleData* boundaryData,
        const SolverConfiguration* configuration
    );
    ~Solver();

    void Bind() const;
    void Advance(float dt);

private:
    inline void computeNeighborhoods(unsigned char resID);
    inline void computeDensities(unsigned char resID);
    inline void computeAccelerations(unsigned char resID);
    inline void integrate(unsigned char resID, float timeStep);

    SolverConfiguration mConfiguration;
    SPHParticleData* mFluidData[2];
    BoundaryParticleData* mBoundaryData;
};
//------------------------------------------------------------------------------