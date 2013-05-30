//------------------------------------------------------------------------------
//  SSFRendererDepthFragment.glsl
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
out float outIndex;
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

    n.z = sqrt(1.0f - mag2);

    // compute view space pos for this fragment
    vec4 posView = inGeometryData.PosView + uParticleRadius*vec4(n, 1.0f);
    outFragOutput = -posView.z;
    
    if ((inGeometryData.State & 4) == 4)
    {
        0.0f;
    }
    else
    {
        outIndex = float(inGeometryData.VertexID) + 1.0f;
    }
    
    vec4 posClip = uProjMat*posView;
    gl_FragDepth = posClip.z/posClip.w;
}
//------------------------------------------------------------------------------