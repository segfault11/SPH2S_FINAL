//------------------------------------------------------------------------------
//  cuda.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 18.02.13.
//  Copyright (c) 2013. All rights reserved.
//------------------------------------------------------------------------------
#include <iostream>
#include "cuda.h"
//------------------------------------------------------------------------------
CUDA::Timer::Timer ()
    : mState(CT_STOPPED)
{
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
}
//------------------------------------------------------------------------------
CUDA::Timer::~Timer ()
{
    cudaEventDestroy(mStart);
    cudaEventDestroy(mStop);
}
//------------------------------------------------------------------------------
void CUDA::Timer::Start () 
{
    cudaEventRecord(mStart, 0);
    mState = CT_TAKING_TIME;
}
//------------------------------------------------------------------------------
void CUDA::Timer::Stop ()
{
    cudaEventRecord(mStop, 0);
    cudaEventSynchronize(mStop);
    cudaEventElapsedTime(&mTime, mStart, mStop);
    mState = CT_STOPPED;
}
//------------------------------------------------------------------------------
void CUDA::Timer::DumpElapsed () const
{
    if (mState == CT_TAKING_TIME)
    {
        return;
    }

    std::cout << mTime << " [ms] elapsed" << std::endl;
}
//------------------------------------------------------------------------------
float CUDA::Timer::GetElapsed() const
{
    return mTime;
}
//------------------------------------------------------------------------------