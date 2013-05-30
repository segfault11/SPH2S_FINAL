//------------------------------------------------------------------------------
//  SSFRenderer.h
//
//  Implements the Geometry class defined in "SSFRenderer.h"
//------------------------------------------------------------------------------
#include "SSFRenderer.h"
//------------------------------------------------------------------------------
SSFRenderer::Geometry::Geometry(
    const float* vertices, 
    const float* normals,
    unsigned int numFaces
)
:
    mNumFaces(numFaces)
{
    // (1) create a vertex buffer object containing the vertex positions. 
    //     create a vertex buffer object containing the vertex normals.
    // (2) create and vertex array that stores positions and normals
    // (3) configure the vertex array for both buffer objects

    // create buffer object for the vertex positions    
    GL::CreateBufferObject(
        mPositionsVBO, 
        GL_ARRAY_BUFFER, 
        sizeof(float)*3*3*numFaces,
        vertices,
        GL_STATIC_DRAW
    );

    // create buffer object for the vertex normals
    GL::CreateBufferObject(
        mNormalsVBO, 
        GL_ARRAY_BUFFER, 
        sizeof(float)*3*3*numFaces,
        normals,
        GL_STATIC_DRAW
    );

    // create a vertex array object
    glGenVertexArrays(1, &mVAO);
    glBindVertexArray(mVAO);

    // bind the buffer object that stores the vertex positions and configure
    glBindBuffer(GL_ARRAY_BUFFER, mPositionsVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // bind the buffer object that stores the vertex normals and configure
    glBindBuffer(GL_ARRAY_BUFFER, mNormalsVBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0); 
    glEnableVertexAttribArray(1);

    // finished with the vertex array object
    glBindVertexArray(0);

}
//------------------------------------------------------------------------------
SSFRenderer::Geometry::~Geometry ()
{
    glDeleteVertexArrays(1, &mVAO);
    glDeleteBuffers(1, &mPositionsVBO);
    glDeleteBuffers(1, &mNormalsVBO);
}
//------------------------------------------------------------------------------
void SSFRenderer::Geometry::Render() const
{
    glBindVertexArray(mVAO);
    glDrawArrays(GL_TRIANGLES, 0, 3*mNumFaces);
}
//------------------------------------------------------------------------------
SSFRenderer::Geometry* SSFRenderer::Geometry::CreateBox(
    const float3& startPoint, 
    const float3& endPoint
)
{
    //// compute the amount the unit cube needs to be scaled in each direction
    float scale[3];
    scale[0] = (endPoint.x - startPoint.x)/2.0f;
    scale[1] = (endPoint.y - startPoint.y)/2.0f;
    scale[2] = (endPoint.z - startPoint.z)/2.0f;
     
    // compute the vector the quad needs to be translated by
    float translate[3];
    translate[0] = startPoint.x + scale[0]; 
    translate[1] = startPoint.y + scale[1]; 
    translate[2] = startPoint.z + scale[2]; 

    GLfloat vertices[] = {
             // front
             -1.0f, -1.0f,  1.0f,
             1.0f,  -1.0f,  1.0f,
             1.0f,   1.0f,  1.0f,
             
            -1.0f,  -1.0f,  1.0f,
            -1.0f,   1.0f,  1.0f,
             1.0f,   1.0f,  1.0f,
             
             // back
             -1.0f, -1.0f,  -1.0f,
             1.0f,  -1.0f,  -1.0f,
             1.0f,   1.0f,  -1.0f,
             
            -1.0f,  -1.0f,  -1.0f,
            -1.0f,   1.0f,  -1.0f,
             1.0f,   1.0f,  -1.0f,

             // left
             -1.0f, -1.0f,  1.0f,
             -1.0f, -1.0f, -1.0f,
             -1.0f,  1.0f, -1.0f,
             
             -1.0f, -1.0f,  1.0f,
             -1.0f,  1.0f,  1.0f,
             -1.0f,  1.0f, -1.0f,

             // right
             1.0f, -1.0f,  1.0f,
             1.0f, -1.0f, -1.0f,
             1.0f,  1.0f, -1.0f,
             
             1.0f, -1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f, -1.0f,

             // top
             -1.0f, 1.0f,  1.0f,
             -1.0f,  1.0f,  -1.0f,
              1.0f,  1.0f,  -1.0f,
             
            -1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  1.0f,
             1.0f,  1.0f,  -1.0f,

             // bottom
             -1.0f, -1.0f,  1.0f,
             -1.0f, -1.0f,  -1.0f,
             1.0f,  -1.0f,  -1.0f,
             
            -1.0f,  -1.0f,  1.0f,
             1.0f,  -1.0f,  1.0f,
             1.0f,  -1.0f,  -1.0f,

         };

     GLfloat normals[] = {
            // front
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,

            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f,

            // back
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,

            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,
            0.0f, 0.0f, -1.0f,

            // left
            -1.0f,  0.0f,  0.0,
            -1.0f,  0.0f,  0.0,
            -1.0f,  0.0f,  0.0,

            -1.0f,  0.0f,  0.0,
            -1.0f,  0.0f,  0.0,
            -1.0f,  0.0f,  0.0,

            // right
            1.0f,  0.0f,  0.0,
            1.0f,  0.0f,  0.0,
            1.0f,  0.0f,  0.0,

            1.0f,  0.0f,  0.0,
            1.0f,  0.0f,  0.0,
            1.0f,  0.0f,  0.0,

            // top
            0.0f,  0.0f,  1.0,
            0.0f,  0.0f,  1.0,
            0.0f,  0.0f,  1.0,

            0.0f,  0.0f,  1.0,
            0.0f,  0.0f,  1.0,
            0.0f,  0.0f,  1.0,

            // top
            0.0f,  0.0f, -1.0,
            0.0f,  0.0f, -1.0,
            0.0f,  0.0f, -1.0,

            0.0f,  0.0f, -1.0,
            0.0f,  0.0f, -1.0,
            0.0f,  0.0f, -1.0
        };


    // scale the unit cube
    for (unsigned int i = 0; i < sizeof(vertices)/(sizeof(float)*3);  i++)
    {
        vertices[3*i + 0] *= scale[0]; 
        vertices[3*i + 1] *= scale[1]; 
        vertices[3*i + 2] *= scale[2]; 
    }

    // translate the unit cube
    for (unsigned int i = 0; i < sizeof(vertices)/(sizeof(float)*3);  i++)
    {
        vertices[3*i + 0] += translate[0]; 
        vertices[3*i + 1] += translate[1]; 
        vertices[3*i + 2] += translate[2]; 
    }

    Geometry* geometry = new Geometry(
            vertices, 
            normals, 
            sizeof(vertices)/(sizeof(float)*3*3)
        );
    return geometry;  
}
//------------------------------------------------------------------------------