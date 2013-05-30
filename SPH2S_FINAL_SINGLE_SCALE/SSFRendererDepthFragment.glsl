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
} 
inGeometryData;
//------------------------------------------------------------------------------
out vec4 outFragOutput;
//------------------------------------------------------------------------------
void main ()
{
    vec3 n;
    n.x = 2.0f*inGeometryData.TexCoord.x - 1.0f;
    n.y = 2.0f*inGeometryData.TexCoord.y - 1.0f;
    float mag2 = dot(n.xy, n.xy);
    
    if (mag2 > 1.0f)
    {
        discard;
    }

    n.z = sqrt(1.0f - mag2);

    // compute view space pos for this fragment
    vec4 posView = inGeometryData.PosView + uParticleRadius*vec4(n, 1.0f);
    outFragOutput = vec4(-posView.z, 0.0f, 0.0f, 0.0f);
    vec4 posClip = uProjMat*posView;
    gl_FragDepth = 0.5f*(posClip.z/posClip.w + 1.0f);
}
//------------------------------------------------------------------------------