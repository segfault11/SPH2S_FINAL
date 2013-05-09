//------------------------------------------------------------------------------
//  BoxRenderer.h
//------------------------------------------------------------------------------
#include "BoxRenderer.h"
//------------------------------------------------------------------------------
#pragma once
//------------------------------------------------------------------------------
BoxRenderer::BoxRenderer (const float3& start, const float3& end)
{
    //==========================================================================
    // Create and configure glsl object
    //==========================================================================    
    mProgram = glCreateProgram();
    GL::AttachShader(mProgram, "BoxRendererVertex.glsl", GL_VERTEX_SHADER);
    GL::AttachShader(mProgram, "BoxRendererFragment.glsl", GL_FRAGMENT_SHADER);
    GL::BindAttribLocation(mProgram, "iPositions", 0);
    GL::BindFragDataLocation(mProgram, "oFragColor", 0);
    GL::LinkProgram(mProgram);

    //==========================================================================
    // Create and configure VAO
    //==========================================================================
    float data[72];
    memset(data, 0 , 72*sizeof(float));
    
    // 0th segment    
    data[0*6 + 0] = start.x; 
    data[0*6 + 1] = start.y; 
    data[0*6 + 2] = start.z; 
    data[0*6 + 3] = end.x; 
    data[0*6 + 4] = start.y; 
    data[0*6 + 5] = start.z; 

    // 1st segment    
    data[1*6 + 0] = start.x; 
    data[1*6 + 1] = start.y; 
    data[1*6 + 2] = start.z; 
    data[1*6 + 3] = start.x; 
    data[1*6 + 4] = end.y; 
    data[1*6 + 5] = start.z; 

    // 2nd segment    
    data[2*6 + 0] = start.x; 
    data[2*6 + 1] = start.y; 
    data[2*6 + 2] = start.z; 
    data[2*6 + 3] = start.x; 
    data[2*6 + 4] = start.y; 
    data[2*6 + 5] = end.z; 

    // 3rd segment    
    data[3*6 + 0] = end.x; 
    data[3*6 + 1] = end.y; 
    data[3*6 + 2] = end.z; 
    data[3*6 + 3] = start.x; 
    data[3*6 + 4] = end.y; 
    data[3*6 + 5] = end.z; 

    // 4th segment    
    data[4*6 + 0] = end.x; 
    data[4*6 + 1] = end.y; 
    data[4*6 + 2] = end.z; 
    data[4*6 + 3] = end.x; 
    data[4*6 + 4] = start.y; 
    data[4*6 + 5] = end.z; 

    // 5th segment    
    data[5*6 + 0] = end.x; 
    data[5*6 + 1] = end.y; 
    data[5*6 + 2] = end.z; 
    data[5*6 + 3] = end.x; 
    data[5*6 + 4] = end.y; 
    data[5*6 + 5] = start.z; 

    // 6th segment    
    data[6*6 + 0] = start.x; 
    data[6*6 + 1] = end.y; 
    data[6*6 + 2] = start.z; 
    data[6*6 + 3] = end.x; 
    data[6*6 + 4] = end.y; 
    data[6*6 + 5] = start.z;

    // 7th segment    
    data[7*6 + 0] = start.x; 
    data[7*6 + 1] = end.y; 
    data[7*6 + 2] = start.z; 
    data[7*6 + 3] = start.x; 
    data[7*6 + 4] = end.y; 
    data[7*6 + 5] = end.z;

    // 8fth segment    
    data[8*6 + 0] = start.x; 
    data[8*6 + 1] = end.y; 
    data[8*6 + 2] = end.z; 
    data[8*6 + 3] = start.x; 
    data[8*6 + 4] = start.y; 
    data[8*6 + 5] = end.z;
    
    // 9fth segment    
    data[9*6 + 0] = start.x; 
    data[9*6 + 1] = start.y; 
    data[9*6 + 2] = end.z;
    data[9*6 + 3] = end.x; 
    data[9*6 + 4] = start.y; 
    data[9*6 + 5] = end.z;

    // 10th segment
    data[10*6 + 0] = end.x; 
    data[10*6 + 1] = start.y; 
    data[10*6 + 2] = start.z; 
    data[10*6 + 3] = end.x; 
    data[10*6 + 4] = end.y; 
    data[10*6 + 5] = start.z; 

    // 11th segment
    data[11*6 + 0] = end.x; 
    data[11*6 + 1] = start.y; 
    data[11*6 + 2] = start.z; 
    data[11*6 + 3] = end.x; 
    data[11*6 + 4] = start.y; 
    data[11*6 + 5] = end.z; 

    GL::CreateBufferObject(mBoxVBO, GL_ARRAY_BUFFER, sizeof(float)*72,
        data, GL_STATIC_DRAW);

    glGenVertexArrays(1, &mBoxVAO);
    glBindVertexArray(mBoxVAO); 
    glBindBuffer(GL_ARRAY_BUFFER, mBoxVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
}
//------------------------------------------------------------------------------
BoxRenderer::~BoxRenderer ()
{
    glDeleteVertexArrays(1, &mBoxVAO);
    glDeleteBuffers(1, &mBoxVBO);
}
//------------------------------------------------------------------------------
void BoxRenderer::SetCamera (const GL::Camera& camera)
{
    glUseProgram(mProgram);
    GLfloat projMat[16];
    GLfloat viewMat[16];
    GL::Camera::ComputeProjectionMatrix(projMat, camera);
    GL::Camera::ComputeViewMatrix(viewMat, camera);
    GLint loc = glGetUniformLocation(mProgram, "uProjMat");
    glUniformMatrix4fv(loc, 1, false, projMat);
    loc = glGetUniformLocation(mProgram, "uViewMat");
    glUniformMatrix4fv(loc, 1, false, viewMat);
}
//------------------------------------------------------------------------------
void BoxRenderer::Render () const
{
    glUseProgram(mProgram);
    glBindVertexArray(mBoxVAO);
    glDrawArrays(GL_LINES, 0, 36);
}
//------------------------------------------------------------------------------

