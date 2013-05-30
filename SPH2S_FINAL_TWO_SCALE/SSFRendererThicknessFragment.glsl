//------------------------------------------------------------------------------
//  SSFRendererThicknessFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform float uParticleRadius;
//------------------------------------------------------------------------------
in GeometryData 
{
	vec2 TexCoord;
    vec4 PosView;
    flat int VertexID;
    flat int State;
} 
inGeometryData;
//------------------------------------------------------------------------------
out float outFragOutput;
//------------------------------------------------------------------------------
void main ()
{
    vec3 n;
    n.x = 2.0f*inGeometryData.TexCoord.x - 1.0f;
    n.y = 2.0f*inGeometryData.TexCoord.y - 1.0f;
    float mag2 = dot(n.xy, n.xy);
    
    if (mag2 > 1.0f || inGeometryData.State == 1)
    {
        discard;
    }

    outFragOutput = vec4(2.0f*uParticleRadius, 0.0f, 0.0f, 0.0f);
}
//------------------------------------------------------------------------------