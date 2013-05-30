//------------------------------------------------------------------------------
//  SSFRendererSphereVertex.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
in vec3 inPosition;
in int inState;
//------------------------------------------------------------------------------
out VertexData 
{
    float SphereRadiusX;    // sphere radius (xDirection) in NDC space
    float SphereRadiusY;    // sphere radius (yDirection) in NDC space
    vec4 PosView;           // position of the particle in view space
    flat int VertexID;
    flat int State;
}
outVertexData;
//------------------------------------------------------------------------------
void main ()
{
    vec4 posView = uViewMat*vec4(inPosition, 1.0f);
    outVertexData.PosView = posView;
    outVertexData.SphereRadiusX = uProjMat[0][0]*uParticleRadius/-posView.z;
    outVertexData.SphereRadiusY = uProjMat[1][1]*uParticleRadius/-posView.z;
    outVertexData.VertexID = gl_VertexID;
    outVertexData.State = inState;
    gl_Position = uProjMat*posView/-posView.z;
}
//------------------------------------------------------------------------------
