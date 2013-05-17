//-----------------------------------------------------------------------------
//  VideoWriter.h
//  FastTurbulentFluids
//
//  Created by Arno in Wolde Lübke on 19.02.13.
//  Copyright (c) 2013. All rights reserved.
//-----------------------------------------------------------------------------
#pragma once 

#include <string>
//#include "../stdutil.h"
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include "../OpenGL/OpenGL.h"

class VideoWriter
{
   // DECL_DEFAULTS (VideoWriter)

public:
    VideoWriter (const std::string& filename, unsigned int width, 
        unsigned int height);
    ~VideoWriter ();

    void SaveScreenshot (const std::string& filename) const;    
    void CaptureFrame () const;
    void Save (const std::string& filename) const;
    
private:
    std::string mFilename;
    unsigned int mWidth;
    unsigned int mHeight;

    CvVideoWriter* mWriter;
    IplImage* mCurrentFrame;

};