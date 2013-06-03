//------------------------------------------------------------------------------
//  main.cpp
//------------------------------------------------------------------------------
#include <iostream>
#include <cstdlib>
#include <memory>
#include "OpenGL\OpenGL.h"
#include "cuda.h"
#include "Renderer.h"
#include "SSFRenderer.h"
#include "ParticleData.h"
#include "Solver.h" 
#include "BoxRenderer.h"
#include "VideoWriter\VideoWriter.h"
//------------------------------------------------------------------------------
#define PI 3.14159265358979323846
#define WIDTH  1024
#define HEIGHT 768
//------------------------------------------------------------------------------
static ParticleData* gsParticleData;
static ParticleData* gsParticleDataHigh;
static ParticleData* gsBoundaryParticles;
static GL::Camera* gsCamera;
static Renderer* gsRenderer;
static SSFRenderer* gsSSFRenderer;
static Renderer* gsRendererHigh;
static Renderer* gsBoundaryRenderer;
static BoxRenderer* gsBoxRenderer;
static Solver* gsSolver;
static VideoWriter* gsVideoWriter;
static float gsDAngY = 0.0f;
//------------------------------------------------------------------------------
static void display();
static void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void mouseMotion(int x, int y);
static void initGL();
static void initSim();
static void tearDownSim();
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
    //gsRenderer->SetCamera(*gsCamera);
    //gsRenderer->Render();
    //gsRendererHigh->SetCamera(*gsCamera);
    //gsRendererHigh->Render();
    //gsBoxRenderer->SetCamera(*gsCamera);
    //gsBoxRenderer->Render();
    gsSSFRenderer->SetCamera(*gsCamera);
    gsSSFRenderer->Render();
    glFlush();
    glutSwapBuffers();
    glutPostRedisplay();
    gsCamera->RotateAroundFocus(0.0f, gsDAngY, 0.0f);

    static int i = 0;

    if (i % 5 == 0)
    {
        gsVideoWriter->CaptureFrame();
    }

    if (i == 2000)
    {
        tearDownSim();
        exit(0);
    }

    i++;
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
    // INIT PARTICLE DATA
    Grid particleGrid = Grid::MakeGrid(
            make_float3(0.02f, 0.04f, -0.05f),
            make_float3(0.35f, 0.75f, 0.45f),
            0.0075f
        );
    gsParticleData = ParticleData::CreateParticleBox(particleGrid);
    gsParticleDataHigh = new ParticleData(gsParticleData->MaxParticles*8);
    gsParticleDataHigh->NumParticles = 0;    
    
    // INIT BOUNDARY PARTICLES
    Grid boundaryGridA = Grid::MakeGrid(
            make_float3(0.0f, 0.0f, -0.1f),
            make_float3(1.6f, 1.5f, 0.5f),
            0.015f
        );
    Grid boundaryGridB = Grid::MakeGrid(
            make_float3(0.7f, 0.0f, 0.22f),
            make_float3(0.75f, 1.5f, 0.25f),
            0.015f
        );
    ParticleData* boundaryA = ParticleData::CreateParticleBoxCanvas(
        boundaryGridA, 
        5
    );
    ParticleData* boundaryB = ParticleData::CreateParticleBoxCanvas(
        boundaryGridB, 
        5
    );
    gsBoundaryParticles = ParticleData::Union(boundaryA, boundaryB);
    delete boundaryA;
    delete boundaryB;

    // INIT CAMERA
    gsCamera = new GL::Camera(
        GL::Vector3f(1.6f, 0.75f, 2.0f),
        GL::Vector3f(0.8f, 0.75f, 0.2f),
        GL::Vector3f(0.0f, 1.0f, 0.0f),
        static_cast<float>(WIDTH)/static_cast<float>(HEIGHT),
        60.0f,
        0.1f,
        10.0f
    );

    // INIT RENDERER
    RendererConfig rendererConfig(
            make_float3(0.0f, 0.5f, 1.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            0.1f,
            0.5f,
            0.5f,
            0.008f
        );
    gsRenderer = new Renderer(gsParticleData, rendererConfig);
    gsRenderer->SetCamera(*gsCamera);
    
    // INIT HIGH RES RENDERER
    RendererConfig rendererConfigHigh(
            make_float3(1.0f, 0.5f, 0.0f),
            make_float3(1.0f, 1.0f, 1.0f),
            0.1f,
            0.5f,
            0.5f,
            0.0085f
        );    
    gsRendererHigh = new Renderer(gsParticleDataHigh, rendererConfigHigh);
    
    // INIT BOUNDARY RENDERER
    gsBoundaryRenderer = new Renderer(gsBoundaryParticles, rendererConfig);
    gsBoundaryRenderer->SetCamera(*gsCamera);
    gsBoxRenderer = new BoxRenderer(
            make_float3(0.0f, 0.0f, -0.1f),
            make_float3(1.5f, 1.5f, 0.5f)
        );
    gsBoxRenderer->SetCamera(*gsCamera);

    // INIT SOLVER
    SolverConfiguration config = SolverConfiguration::MakeConfiguration(
            make_float3(-0.1f, -0.1f, -0.2f),
            make_float3(1.7f, 1.6f, 0.6f),
            Grid::ComputeVolume(particleGrid),
            gsParticleData->MaxParticles,
            35,                      // avg. particle neighbors
            1000.0f,                 // rest density
            30.0f,                   // bulk modulus
            5.0f,                    // viscosity
            88.1472f,                // speed of sound in fluid
            0.8f,                    // tension coefficient
            0.0125f                  // blend increment
        );

    std::cout << gsParticleData->MaxParticles << std::endl;
    //std::cout << std::powf(3.0f/4.0f*config.FluidParticleMass[0]/config.RestDensity*1.0f/M_PI, 1.0f/3.0f) << std::endl;
    std::system("pause");
    
    
    // CREATE A SOLVER AND SET IT ACTIVE
    gsSolver = new Solver(
            gsParticleData, 
            gsParticleDataHigh, 
            gsBoundaryParticles, 
            &config
        ); 
    gsSolver->Bind();
   
    gsSSFRenderer = new SSFRenderer(
            gsParticleData, 
            gsParticleDataHigh, 
            WIDTH, 
            HEIGHT, 
            0.015*1.2f/2.0f
        );

    gsVideoWriter = new VideoWriter("video_high.avi", WIDTH, HEIGHT);
    gsSSFRenderer->AddBox(
            make_float3(0.62f, -0.0f, 0.138f),
            make_float3(0.83f, 1.5f, 0.332)
    );
}
//------------------------------------------------------------------------------
void tearDownSim()
{
    delete gsParticleData;
    delete gsParticleDataHigh;
    delete gsBoundaryParticles;
    delete gsCamera;
    delete gsRenderer;
    delete gsRendererHigh;
    delete gsSolver;
    delete gsSSFRenderer;
    delete gsVideoWriter;
}
//------------------------------------------------------------------------------