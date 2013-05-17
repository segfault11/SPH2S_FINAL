//------------------------------------------------------------------------------
//  SSFRendererDepthFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform sampler2D uDepthSampler;
uniform sampler2D uThicknessSampler;
uniform mat4 uProjMat;
uniform float uTexSizeX;
uniform float uTexSizeY;
//------------------------------------------------------------------------------
in VertexData
{
    vec2 TexCoord;
}
inVertexData;
//------------------------------------------------------------------------------
out vec4 outFragOut;
//------------------------------------------------------------------------------
vec3 getViewPos(vec2 texCoord)
{
    vec3 viewPos = vec3(1.0f, 1.0f, 1.0f);
    float linearDepth = texture(uDepthSampler, texCoord).r;
    viewPos.x = (2.0f*texCoord.x - 1.0f + uProjMat[0][2])*
        linearDepth/uProjMat[0][0];
    viewPos.y = (2.0f*texCoord.y - 1.0f + uProjMat[1][2])*
        linearDepth/uProjMat[1][1];
    viewPos.z = -linearDepth;
    return viewPos;
} 
//------------------------------------------------------------------------------
void main()
{
    vec3 colorFluid = vec3(0.0f, 0.5f, 0.8f);
    vec3 colorEnv = vec3(1.0f, 1.0f, 1.0f);

    //--------------------------------------------------------------------------
    // reconstruct the view space normal of the fragment
    //--------------------------------------------------------------------------

    vec3 posView = getViewPos(inVertexData.TexCoord);

    if (posView.z == 0.0f)
    {
        discard;
    }

    vec3 ddx = getViewPos(inVertexData.TexCoord + vec2(uTexSizeX, 0.0f)) - 
        posView;
    vec3 ddx2 = -getViewPos(inVertexData.TexCoord - vec2(uTexSizeX, 0.0f)) + 
        posView;
    vec3 ddy = getViewPos(inVertexData.TexCoord + vec2(0.0f, uTexSizeY)) - 
        posView;
    vec3 ddy2 = -getViewPos(inVertexData.TexCoord - vec2(0.0f, uTexSizeY)) + 
        posView;

    if (abs(ddx.z) > abs(ddx2.z))
    {
        ddx = ddx2;
    }

    if (abs(ddy.z) > abs(ddy.z))
    {
        ddy = ddy2;
    }

    vec3 n = cross(ddx, ddy);
    n = normalize(n);

    //--------------------------------------------------------------------------
    // get thickness of the fragment
    //--------------------------------------------------------------------------
    float t = texture(uThicknessSampler, inVertexData.TexCoord).r;

    float tc = exp(-0.5f*t);


    //--------------------------------------------------------------------------
    // compute fresnel reflectance (schlicks approx)
    //--------------------------------------------------------------------------
    vec3 v = normalize(-posView);
    float a = 1.0f - max(0.0f, dot(v, n));
    float b = a*a*a*a*a;
    float rf = 0.02f + 0.98f*b; // rf0 for water is = 0.02f 




    outFragOut = vec4((1 - rf)*mix(colorFluid, colorEnv, tc) + rf*colorEnv, 1.0f);
}
//------------------------------------------------------------------------------
