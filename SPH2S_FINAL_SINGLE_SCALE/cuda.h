//------------------------------------------------------------------------------
//  cuda.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 12.02.13.
//  Copyright (c) 2013. All rights reserved.
//------------------------------------------------------------------------------
#pragma once
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>    
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>
#include <thrust\fill.h>
#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include <thrust\for_each.h>
#include <thrust\iterator\zip_iterator.h>
//------------------------------------------------------------------------------
#define CUDA_SAFE_CALL(x) SafeCall(x, __FILE__, __LINE__);
#define CUDA_SAFE_INV(x) SafeInv(x, __FILE__, __LINE__);
//------------------------------------------------------------------------------
inline void SafeInv (cudaError_t err, const char* filename, unsigned int line)
{
    using namespace std;

    if (err == cudaSuccess)
    {
        return;
    }    
    stringstream str(stringstream::in | stringstream::out);

    str << "Exception thrown in FILE: " << filename << " LINE: " << line << endl;
    str << "Error Message: " << cudaGetErrorString(cudaGetLastError()) << endl;
    
    throw runtime_error(str.str());
}
//------------------------------------------------------------------------------
namespace CUDA
{
//------------------------------------------------------------------------------
template<typename T>
inline void Alloc (T** data, unsigned int size)
{
    CUDA_SAFE_CALL( cudaMalloc(reinterpret_cast<void**>(data), sizeof(T)*size) );
}
//------------------------------------------------------------------------------
template<typename T>
inline void Free (T** data)
{
    if (*data != NULL) 
    {
        cudaFree(*data);
        *data = NULL;
    }
}
//------------------------------------------------------------------------------
template<typename T>
inline void Fill (T* data, unsigned int n, T val)
{
    T* hData = new T[n];

    for (unsigned int i = 0; i < n; i++)
    {
        hData[i] = val;
    }

    cudaMemcpy(data, hData, sizeof(T)*n, cudaMemcpyHostToDevice);

    delete[] hData;
}
//------------------------------------------------------------------------------
template<typename T>
inline void Fill (T* data, unsigned int n, T start, T increment)
{
    T* hData = new T[n];

    for (unsigned int i = 0; i < n; i++)
    {
        hData[i] = start + static_cast<T>(i)*increment;
    }

    cudaMemcpy(data, hData, sizeof(T)*n, cudaMemcpyHostToDevice);

    delete[] hData;
}
//------------------------------------------------------------------------------
template<typename T>
inline void Memset (void *devPtr, int value, size_t count)
{
    CUDA_SAFE_CALL( cudaMemset(devPtr, value, sizeof(T)*count) );
}
//------------------------------------------------------------------------------
template<typename T>
inline void Memcpy (
    T *dst, 
    const T *src, 
    size_t count, 
    enum cudaMemcpyKind kind
)
{
    
    CUDA_SAFE_CALL( cudaMemcpy(dst, src, sizeof(T)*count, kind));
}
//------------------------------------------------------------------------------
template<typename T>
inline void MemcpyToSymbol (const T* symbol, const T* src, unsigned int n)
{
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(symbol, src, sizeof(T)*n) );
}
//------------------------------------------------------------------------------
inline void SafeCall (cudaError_t err, const char* filename, unsigned int line)
{
    using namespace std;

    if (err == cudaSuccess)
    {
        return;
    }    
    stringstream str(stringstream::in | stringstream::out);

    str << "Exception thrown in FILE: " << filename << " LINE: " << line << endl;
    str << "Error Message: " << cudaGetErrorString(cudaGetLastError()) << endl;
    
    throw runtime_error(str.str());
}
//------------------------------------------------------------------------------
template<typename T>
inline void SafeFree (T** ptr ) 
{
    if (*ptr != NULL) {
        cudaFree(*ptr);
        *ptr = NULL;
    }
}
//------------------------------------------------------------------------------
template<typename T> 
inline void DumpArray (
    T* arr, unsigned int numElements, 
    unsigned int offset = 0, 
    unsigned int stride = 1, 
    unsigned int pauseAfter = 0
)
{
    T* hostData = new T[numElements];
    
    CUDA_SAFE_CALL( cudaMemcpy(hostData, arr, sizeof(T)*numElements,
        cudaMemcpyDeviceToHost) );
    
    for (unsigned int i = 0; i < numElements; i++)
    {
        std::cout << i << " " << hostData[i*stride + offset] << std::endl;

        if (pauseAfter != 0)
        {
            if ((i % pauseAfter) == 0)
            {
                system("pause");
            }
        }
    }
    
    delete[] hostData;
}
//------------------------------------------------------------------------------
template<typename T> 
inline void DumpArrayNE (T* arr, unsigned int numElements, T ne)
{
    T* hostData = new T[numElements];
    
    CUDA_SAFE_CALL( cudaMemcpy(hostData, arr, sizeof(T)*numElements,
        cudaMemcpyDeviceToHost) );
    
    for (unsigned int i = 0; i < numElements; i++)
    {
        if (hostData[i] != ne)
        {
            std::cout << i << " " << hostData[i] << std::endl;
        }
    }
    
    delete[] hostData;
}
//------------------------------------------------------------------------------
class Timer
{
    enum 
    {
        CT_TAKING_TIME,
        CT_STOPPED
    };

public:
    Timer();
    ~Timer();
    void Start();
    void Stop();
    void DumpElapsed() const;

private:
    cudaEvent_t mStart;
    cudaEvent_t mStop;
    float mTime;
    int mState;
};
//------------------------------------------------------------------------------
namespace GL
{
//------------------------------------------------------------------------------
inline void RegisterImage(
    struct cudaGraphicsResource** resource,
    GLuint image,
    GLenum target,
    unsigned int flags
)
{
    CUDA_SAFE_CALL( 
        cudaGraphicsGLRegisterImage(
            resource, 
            image, 
            target, 
            flags
        ) 
    ); 
}
//------------------------------------------------------------------------------
inline void RegisterBuffer (
    struct cudaGraphicsResource** resource, 
    GLuint buffer, 
    unsigned int flags
)
{
    CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer(resource, buffer, flags) );
}
//------------------------------------------------------------------------------
inline void MapResources (int count, cudaGraphicsResource_t* resources)
{
    CUDA_SAFE_CALL (cudaGraphicsMapResources(count, resources) );
}
//------------------------------------------------------------------------------
template<typename T>
inline void GetMappedPointer (
    T** devPtr, 
    cudaGraphicsResource_t resource
)
{
    size_t nBytes;
    CUDA_SAFE_CALL(
        cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(devPtr),
            &nBytes,
            resource
        )
    );
}
//------------------------------------------------------------------------------
inline void UnmapResources (int count, cudaGraphicsResource_t* resources)
{
    CUDA_SAFE_CALL( 
        cudaGraphicsUnmapResources(count, resources) 
    );  
        
}
//------------------------------------------------------------------------------
}   // end of namespace GL
//------------------------------------------------------------------------------
}   // end of namespace CUDA
//------------------------------------------------------------------------------

