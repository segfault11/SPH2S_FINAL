//------------------------------------------------------------------------------
#include "Renderer.h"
#include <iostream>
//------------------------------------------------------------------------------
Renderer::Renderer (const ParticleData* data, const RendererConfig& config)
: mParticleData(data)
{
    //==========================================================================
    // Create and configure glsl object
    //==========================================================================    
    mProgram = glCreateProgram();
    GL::AttachShader(mProgram, "RendererVertex.glsl", GL_VERTEX_SHADER);
    GL::AttachShader(mProgram, "RendererGeometry.glsl", GL_GEOMETRY_SHADER);
    GL::AttachShader(mProgram, "RendererFragment.glsl", GL_FRAGMENT_SHADER);
    GL::BindAttribLocation(mProgram, "inPositions", 0);
    GL::BindAttribLocation(mProgram, "inColorVal", 1);
    GL::BindFragDataLocation(mProgram, "oFragColor", 0);
    GL::LinkProgram(mProgram);

    // set programs uniforms
    glUseProgram(mProgram);
    GLint loc = glGetUniformLocation(mProgram, "uLightDir");
    glUniform3fv(loc, 1, reinterpret_cast<const float*>(&config.LightDir));
    loc = glGetUniformLocation(mProgram, "uAmbientCoefficient");
    glUniform1f(loc, config.AmbientCoefficient);
    loc = glGetUniformLocation(mProgram, "uDiffuseCoefficient");
    glUniform1f(loc, config.DiffuseCoefficient);
    loc = glGetUniformLocation(mProgram, "uSpecularCoefficient");
    glUniform1f(loc, config.SpecularCoefficient);
    loc = glGetUniformLocation(mProgram, "uParticleRadius");
    glUniform1f(loc, config.ParticleRadius);
    
    //==========================================================================
    // Create and configure VAO
    //==========================================================================
    glGenVertexArrays(1, &mParticleDataVAO);
    glBindVertexArray(mParticleDataVAO);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleData->PositionsVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, mParticleData->ColorValuesVBO);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);
}
//------------------------------------------------------------------------------
Renderer::~Renderer ()
{
    glDeleteProgram(mProgram);
    glDeleteVertexArrays(1, &mParticleDataVAO);
}
//------------------------------------------------------------------------------
void Renderer::SetCamera (const GL::Camera& camera)
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
void Renderer::Render () const
{
    glUseProgram(mProgram);
    glBindVertexArray(mParticleDataVAO);
    glDrawArrays(GL_POINTS, 0, mParticleData->NumParticles);
//    glDrawElements(GL_POINTS, mParticleData->NumParticles, GL_UNSIGNED_INT, 0);
}
//------------------------------------------------------------------------------
