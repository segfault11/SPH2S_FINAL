//------------------------------------------------------------------------------
//  main.cpp
//------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
#include <memory>
#include "OpenGL\OpenGL.h"
#include "cuda.h"
#include "Renderer.h"
#include "ParticleData.h"
#include "Solver.h" 
#include "BoxRenderer.h"
//------------------------------------------------------------------------------
#define PI 3.14159265358979323846
#define WIDTH  1024
#define HEIGHT 768
//------------------------------------------------------------------------------
ParticleData* gsParticleData;
ParticleData* gsParticleDataHigh;
ParticleData* gsBoundaryParticles;
GL::Camera* gsCamera;
Renderer* gsRenderer;
Renderer* gsRendererHigh;
Renderer* gsBoundaryRenderer;
BoxRenderer* gsBoxRenderer;
Solver* gsSolver;

static float gsDAngY = 0.0f;
//------------------------------------------------------------------------------
static void display ();
static void keyboard (unsigned char key, int x, int y);
void mouse (int button, int state, int x, int y);
void mouseMotion (int x, int y);
static void initGL ();
static void initSim ();
static void tearDownSim ();
//------------------------------------------------------------------------------
int main (int argc, char* argv[])
{  
    cudaGLSetGLDevice(0); 
    glutInit(&argc, argv);
    glutInitContextVersion(3, 3);
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutCreateWindow("");
	glewExperimental = TRUE;
	glewInit();
    initGL();
    initSim();
	glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutMotionFunc(mouseMotion);
	glutKeyboardFunc(keyboard);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, 
        GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutMainLoop();
    return 0;
}
//------------------------------------------------------------------------------
void display () 
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    gsSolver->Advance(0.001f);
    gsRenderer->SetCamera(*gsCamera);
    gsRenderer->Render();
    gsRendererHigh->SetCamera(*gsCamera);
    gsRendererHigh->Render();
    gsBoxRenderer->SetCamera(*gsCamera);
    gsBoxRenderer->Render();
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    gsCamera->RotateAroundFocus(0.0f, gsDAngY, 0.0f);
}
//------------------------------------------------------------------------------
void mouse (int button, int state, int x, int y)
{
    static int lastX = 0;
    static bool first = true;
    int dX = 0;

    switch (state)
    {
        case GLUT_DOWN:
            dX = x - lastX;
            lastX = x;
            
            if (first)
            {
                first = false;
                return;
            }

            gsDAngY = 0.01f;
            return;

        case GLUT_UP:
            first = true;
            gsDAngY = 0.0f;
            return;
    };
} 
//------------------------------------------------------------------------------
void mouseMotion (int x, int y)
{
   
}
//------------------------------------------------------------------------------
void keyboard (unsigned char key, int x, int y)
{
    switch (key) 
    {
        case 's':
            gsCamera->RotateAroundFocus(0.0f, 0.1f, 0.0f);
            return;
	    default:
		    return; 
	}
}
//------------------------------------------------------------------------------
void initGL ()
{
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
}
//------------------------------------------------------------------------------
void initSim ()
{
    //--------------------------------------------------------------------------
    //  init particle data
    Grid particleGrid = Grid::MakeGrid(
            make_float3(0.04f, 0.04f, 0.01f),
            make_float3(0.25f, 0.751f, 0.39f),
            0.015f
        );
    gsParticleData = ParticleData::CreateParticleBox(particleGrid);
    Grid particleGridHigh = Grid::MakeGrid(
            make_float3(1.7f, 0.04f, 0.01f),
            make_float3(1.95f, 0.751f, 0.35f),
            0.0075f
        );
    gsParticleDataHigh = ParticleData::CreateParticleBox(particleGridHigh);
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // init boundary particles
    Grid boundaryGrid = Grid::MakeGrid(
            make_float3(0.0f, 0.0f, -0.1f),
            make_float3(2.0f, 1.5f, 0.5f),
            0.015f
        );
    gsBoundaryParticles = ParticleData::CreateParticleBoxCanvas(
            boundaryGrid, 
            5
        );
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    //  init camera
    gsCamera = new GL::Camera(
        GL::Vector3f(1.0f, 0.75f, 2.0f),
        GL::Vector3f(1.0f, 0.75f, 0.2f),
        GL::Vector3f(0.0f, 1.0f, 0.0f),
        static_cast<float>(WIDTH)/static_cast<float>(HEIGHT),
        60.0f,
        0.1f,
        10.0f
    );
    //--------------------------------------------------------------------------

    //  init Renderer
    RendererConfig rendererConfig(
            make_float3(1.0f, 1.0f, 1.0f),
            0.1f,
            0.5f,
            0.5f,
            0.017f
        );
    gsRenderer = new Renderer(gsParticleData, rendererConfig);
    gsRenderer->SetCamera(*gsCamera);
    
    // init highres particle rendere
    RendererConfig rendererConfigHigh(
            make_float3(1.0f, 1.0f, 1.0f),
            0.1f,
            0.5f,
            0.5f,
            0.0085f
        );    
    gsRendererHigh = new Renderer(gsParticleDataHigh, rendererConfigHigh);
    
    // init boundary renderer
    gsBoundaryRenderer = new Renderer(gsBoundaryParticles, rendererConfig);
    gsBoundaryRenderer->SetCamera(*gsCamera);
    gsBoxRenderer = new BoxRenderer(
            make_float3(0.0f, 0.0f, -0.1f),
            make_float3(2.0f, 1.6f, 0.5f)
        );
    gsBoxRenderer->SetCamera(*gsCamera);

    // init Solver
    SolverConfiguration config = SolverConfiguration::MakeConfiguration(
            make_float3(-0.1f, -0.1f, -0.3f),
            make_float3(2.1f, 1.4f, 0.6f),
            Grid::ComputeVolume(particleGrid),
            gsParticleData->MaxParticles,
            35,                      // avg. particle neighbors
            1000.0f,                 // rest density
            30.0f,                   // bulk modulus
            5.0f,                    // viscosity
            88.1472f,                // speed of sound in fluid
            0.8f                     // tension coefficient
        );
    std::cout << std::powf(3.0f/4.0f*Grid::ComputeVolume(particleGrid)/gsParticleData->MaxParticles*1.0f/M_PI, 1.0f/3.0f) << std::endl;

    // create solver and set it active
    gsSolver = new Solver(
            gsParticleData, 
            gsParticleDataHigh, 
            gsBoundaryParticles, 
            &config
        ); 
    gsSolver->Bind();
    //--------------------------------------------------------------------------
}
//------------------------------------------------------------------------------
void tearDownSim ()
{
    delete gsParticleData;
    delete gsParticleDataHigh;
    delete gsBoundaryParticles;
    delete gsCamera;
    delete gsRenderer;
    delete gsRendererHigh;
    delete gsSolver;
}
//------------------------------------------------------------------------------