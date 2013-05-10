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
        float blendIncrement
    )
    {
        SolverConfiguration config;

        config.EffectiveRadius[0] = std::powf(
                0.75f*volume*static_cast<float>(avgParticles)/
                (M_PI*static_cast<float>(numParticles)), 
                1.0f/3.0f
            ); 
        config.EffectiveRadius[1] = config.EffectiveRadius[0]/2.0f;
        config.FluidParticleMass[0] = restDensity*volume/
            static_cast<float>(numParticles);
        config.FluidParticleMass[1] = config.FluidParticleMass[0]/8.0f;
        config.BoundaryParticleMass = config.FluidParticleMass[0];
        config.RestDensity = restDensity;
        config.Grid[0] = Grid::MakeGrid(origin, end, config.EffectiveRadius[0]);
        config.Grid[1] = Grid::MakeGrid(origin, end, config.EffectiveRadius[1]);
        config.BulkModulus = bulkModulus;
        config.Viscosity = viscosity;
        config.SpeedSound = speedSound;
        config.TensionCoefficient = tensionCoefficient;
        config.BlendIncrement = blendIncrement;
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
    float BlendIncrement;
    ::Grid Grid[2];                // simulation grid
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

        float* dTempPositions;      // temporary positions used for 
                                    // actual computations
        float* dTempVelocities;     // temporary velocities used for 
                                    // actual computations

        float* dBlendCoefficients;
        float* dTempBlendCoefficients;
        unsigned char* dStates;
        unsigned char* dTempStates;

        unsigned int* dActiveIDs;
        unsigned int* dHashs;
        unsigned int* dCellStart;
        unsigned int* dCellEnd;
        unsigned int* dNumParticles; // used to count particles 
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
        ParticleData* fluidDataHigh,
        ParticleData* boundaryData,
        const SolverConfiguration* configuration
    );
    ~Solver();

    void Bind() const;
    void Advance (float dt);

private:
    inline void computeNeighborhoodsLow ();
    inline void computeNeighborhoodsHigh ();
    inline void computeDensitiesLow ();
    inline void computeDensitiesHigh ();
    inline void computeAccelerationsAndUpdateStatesLow ();
    inline void computeAccelerationsAndUpdateStatesHigh ();
    inline void updateSystem (SPHParticleData& fluidData, float timeStep);

    SolverConfiguration mConfiguration;
    SPHParticleData mFluidData;
    SPHParticleData mFluidDataHigh;
    BoundaryParticleData mBoundaryData;
};
//------------------------------------------------------------------------------