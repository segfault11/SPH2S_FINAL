//------------------------------------------------------------------------------
//  RendererVertex.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform float uAmbientCoefficient;
uniform float uDiffuseCoefficient;
uniform float uSpecularCoefficient;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
in vec3 inPosition;
in float inColorVal;
//------------------------------------------------------------------------------
out VertexData 
{
    float SphereRadiusX;    // sphere radius (xDirection) in NDC space
    float SphereRadiusY;    // sphere radius (yDirection) in NDC space
    vec4 PosView;           // position of the particle in view space
    float ColorValue;       
}
outVertexData;
//------------------------------------------------------------------------------
void main ()
{
    float uSphereRadius = 2.0f*0.0062f;
    vec4 posView = uViewMat*vec4(inPosition, 1.0f);
    outVertexData.PosView = posView;
    outVertexData.ColorValue = inColorVal;
    outVertexData.SphereRadiusX = uProjMat[0][0]*uParticleRadius/-posView.z;
    outVertexData.SphereRadiusY = uProjMat[1][1]*uParticleRadius/-posView.z;
    gl_Position = uProjMat*posView/-posView.z;
}
//------------------------------------------------------------------------------
