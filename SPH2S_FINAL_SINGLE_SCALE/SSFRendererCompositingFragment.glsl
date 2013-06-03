//------------------------------------------------------------------------------
//  SSFRendererDepthFragment.glsl
//------------------------------------------------------------------------------
#version 330
//------------------------------------------------------------------------------
uniform sampler2D uDepthSampler;
uniform sampler2D uThicknessSampler;
uniform sampler2D uSceneSampler;
uniform mat4 uProjMat;
uniform float uTexSizeX;
uniform float uTexSizeY;
uniform float uScreenWidth;
uniform float uScreenHeight;
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
    vec3 colorFluid = vec3(0.0f, 0.4f, 1.0f);
    vec3 colorEnv = texture(uSceneSampler, texCoord).rgb;
    vec2 texCoord;
    texCoord.x = gl_FragCoord.x/uScreenWidth;
    texCoord.y = gl_FragCoord.y/uScreenHeight;


    // reconstruct the view space normal of the fragment
    vec3 posView = getViewPos(texCoord);


    // if this fragment does not belong to the fluid, just render the background
    if (posView.z == 0.0f)
    {
        outFragOut = vec4(colorEnv, 1.0f);
        return;
    }

    vec3 ddx = getViewPos(texCoord + vec2(uTexSizeX, 0.0f)) - 
        posView;
    vec3 ddx2 = -getViewPos(texCoord - vec2(uTexSizeX, 0.0f)) + 
        posView;
    vec3 ddy = getViewPos(texCoord + vec2(0.0f, uTexSizeY)) - 
        posView;
    vec3 ddy2 = -getViewPos(texCoord - vec2(0.0f, uTexSizeY)) + 
        posView;

    if (abs(ddx.z) > abs(ddx2.z))
    {
        ddx = ddx2;
    }

    if (abs(ddy.z) > abs(ddy2.z))
    {
        ddy = ddy2;
    }


    //float th = 0.025f;
    //if (abs(ddx.z) > th || abs(ddy.z) > th)
    //{
    //    outFragOut = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    //    return;
    //}

    vec3 n = cross(ddx, ddy);
    n = normalize(n);

    // get thickness of the fragment
    float t = texture(uThicknessSampler, texCoord).r;
    float tc = exp(-2.5f*t);

    // compute fresnel reflectance (schlicks approx)
    vec3 v = normalize(-posView);
    float a = 1.0f - max(0.0f, dot(v, n));
    float b = a*a*a*a*a;
    float rf = 0.02f + 0.98f*b; // rf0 for water is = 0.02f 

    outFragOut = vec4((1 - rf)*mix(colorFluid, colorEnv, tc) + 
        rf*colorEnv, 1.0f);
}
//------------------------------------------------------------------------------
