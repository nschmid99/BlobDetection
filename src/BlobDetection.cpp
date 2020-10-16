
/*
 
 Programmer: Courtney Brown
 Date: c2020
 Notes: Demonstration of blob detection
 Purpose/Description:
 
 This program demonstrates simple blob detection.
 
 Uses:
 
 See brief tutorial here:
 https://www.learnopencv.com/blob-detection-using-opencv-python-c/

 Output/Drawing:
 Draws the result of simple blob detection.
 
 Instructions:
 Copy and paste this code into your cpp file.
 
 Run. Observe the results. Change some of the parameters.
 
 */

//#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

//includes for background subtraction
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/features2d.hpp" //new include for this project

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/Capture.h" //needed for capture
#include "cinder/Log.h" //needed to log errors

//#include "Osc.h" //add to send OSC


#include "CinderOpenCV.h"

#include "Blob.h"

#define SAMPLE_WINDOW_MOD 300
#define MAX_FEATURES 300
#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480

using namespace ci;
using namespace ci::app;
using namespace std;

class BlobDetection : public App {
public:
    void setup() override;
//    void mouseDown( MouseEvent event ) override;
    void keyDown( KeyEvent event ) override;
    
    void update() override;
    void draw() override;
    
protected:
    CaptureRef                 mCapture; //the camera capture object
    gl::TextureRef             mTexture; //current camera frame in opengl format
    
    SurfaceRef                  mSurface; //current camera frame in cinder format
    
    cv::Ptr<cv::SimpleBlobDetector>     mBlobDetector; //the object that does the blob detection
    std::vector<cv::KeyPoint>   mKeyPoints; //the center points of the blobs (current, previous, and before that) & how big they are.
    
    cv::Mat                     mCurFrame, mBackgroundSubtracted; //current frame & frame with background subtracted in OCV format
    cv::Mat                     mSavedFrame; //the frame saved for use for simple background subtraction
    
    cv::Ptr<cv::BackgroundSubtractor> mBackgroundSubtract; //the background subtraction object that does subtractin'
    

    std::vector<Blob>           mBlobs; //the blobs found in the current frame
       std::vector<Blob> mPrevBlobs;
    std::vector<cv::KeyPoint>   mPrevKeyPoints;
    std::vector<int> mMapPrevToCurKeypoints; //maps where points are to where they were in last frame
    
    
    enum BackgroundSubtractionState {NONE=0, OPENCV=2, SAVEDFRAME=3};
    BackgroundSubtractionState mUseBackgroundSubtraction;
    
    void blobDetection(BackgroundSubtractionState useBackground); //does our blob detection
    void createBlobs(); //creates the blob objects for each keypoint

    void blobTracking(); //finds distancec between blobs to track them
    void updateBlobList();//update blobs
    int newBlobID; //the id to assign a new blob.
    int minDist =100;

    
};


void BlobDetection::setup()
{
    //set up our camera
    try {
        mCapture = Capture::create(WINDOW_WIDTH, WINDOW_HEIGHT); //first default camera
        mCapture->start();
    }
    catch( ci::Exception &exc)
    {
        CI_LOG_EXCEPTION( "Failed to init capture ", exc );
    }
    
    //setup the blob detector
    // Setup SimpleBlobDetector parameters.
    cv::SimpleBlobDetector::Params params;
    
    // Change thresholds
    //    params.minThreshold = 10;
    //    params.maxThreshold = 200;
    
    // Filter by Circularity - how circular
    params.filterByCircularity = false;
    params.maxCircularity = 0.2;
    
    // Filter by Convexity -- how convex
    params.filterByConvexity = false;
    params.minConvexity = 0.87;
    
    // Filter by Inertia ?
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;
    
    params.minDistBetweenBlobs = 300.0f;
    
    params.filterByColor = false;
    
    params.filterByArea = true;
    params.minArea = 200.0f;
    params.maxArea = 1000.0f;
    
    //create the blob detector with the above parameters
    mBlobDetector = cv::SimpleBlobDetector::create(params);
    
    //our first available id is 0
    newBlobID = 0;
    
    //use MOG2 -- guassian mixture algorithm to do background subtraction
    mBackgroundSubtract = cv::createBackgroundSubtractorMOG2();
    
    mUseBackgroundSubtraction = BackgroundSubtractionState::NONE;
    mBackgroundSubtracted.data = NULL;
    
}

void BlobDetection::keyDown( KeyEvent event )
{
    if( event.getChar() == '1')
    {
        mUseBackgroundSubtraction = BackgroundSubtractionState::NONE;
    }
    if( event.getChar() == '2')
    {
        mUseBackgroundSubtraction = BackgroundSubtractionState::OPENCV;
    }
    if( event.getChar() == '3')
    {
        mUseBackgroundSubtraction = BackgroundSubtractionState::SAVEDFRAME;
        std::cout << "Saving current frame as background!\n";
        mSavedFrame = mCurFrame;
    }
}

//this function detects the blobs.
void BlobDetection::blobDetection(BackgroundSubtractionState useBackground = BackgroundSubtractionState::NONE)
{
    if(!mSurface) return;
    
    cv::Mat frame;
 
    //using the openCV background subtraction
    if(useBackground == BackgroundSubtractionState::OPENCV)
    {
        mBackgroundSubtract->apply(mCurFrame, mBackgroundSubtracted);
        frame = mBackgroundSubtracted;
    }
    else if( useBackground == BackgroundSubtractionState::SAVEDFRAME )
    {
        if(mSavedFrame.data)
        {
            cv::Mat outFrame;
            
            //use frame-differencing to subtract the background
            cv::GaussianBlur(mCurFrame, outFrame, cv::Size(11,11), 0);
            cv::GaussianBlur(mSavedFrame, mBackgroundSubtracted, cv::Size(11,11), 0);
            cv::absdiff(outFrame, mBackgroundSubtracted, mBackgroundSubtracted);
            
            cv::threshold(mBackgroundSubtracted, mBackgroundSubtracted, 25, 255, cv::THRESH_BINARY);
            
            frame = mBackgroundSubtracted;
        }
        else{
            std::cerr << "No background frame has been saved!\n"; //the way the program is designed, this should happen
        }
    }
    else
    {
        frame = mCurFrame;
    }
    
    //note the parameters: the frame that you would like to detect the blobs in - an input frame
    //& 2nd, the output -- a vector of points, the center points of each blob.
   
    mPrevKeyPoints=mKeyPoints;
    mBlobDetector->detect(frame, mKeyPoints);
    
   
}

void BlobDetection::update()
{
    if(mCapture && mCapture->checkNewFrame()) //is there a new frame???? (& did camera get created?)
    {
        mSurface = mCapture->getSurface();
        
        if(! mTexture)
            mTexture = gl::Texture::create(*mSurface);
        else
            mTexture->update(*mSurface);
    }
    if(!mSurface) return; //we do nothing if there is no new frame
    
    mCurFrame = toOcv(Channel(*mSurface));
    
    
    //update all our blob info
    blobDetection(mUseBackgroundSubtraction);
    blobTracking();
    updateBlobList();
    
    //create blob objects from keypoints
    createBlobs();
   
}

void BlobDetection::createBlobs()
{
    mBlobs.clear(); //create a new list of blobs each time
   // newBlobID = 0; //reset here - since we're not keeping track of blobs yet!
    
    for(int i=0; i<mKeyPoints.size(); i++)
    {
        mBlobs.push_back(Blob(mKeyPoints[i], newBlobID));
        newBlobID++;
    }
   // newBlobID = 0;
}

void BlobDetection::blobTracking(){

    std::vector<int> mindist;
    
   // int saveI;
    int min;
    
    //cycle through and check distance
    for(int j=0; j<mKeyPoints.size(); j++)
    {
        mindist.clear();
    for(int i=0; i<mPrevKeyPoints.size();i++)
    {
        //calculate distance
        int d=ci::distance(fromOcv(mKeyPoints[i].pt), fromOcv(mPrevKeyPoints[j].pt));

        //save distance to a vector
        mindist.push_back(d);

        //remove any false 0's
        mindist.erase(std::remove(mindist.begin(),mindist.end(),0),mindist.end());
        
        //find minimum distance
        min=*min_element(mindist.begin(),mindist.end());
        
        // if minimum distance meets the threshold save the index to map
        if(min<=minDist)
            {
                mMapPrevToCurKeypoints.push_back(i);
                 std::cout<<"in thresh"<<i<<std::endl;
            }

            //if minimum distance exceeds the threshold save -1 to map
             if(min>minDist)
            {
                mMapPrevToCurKeypoints.push_back(-1);
                std::cout<<"exceeds thresh"<<-1<<std::endl;
            }
    }


    }

    
}

void BlobDetection::updateBlobList()
{
    //save blobs to previousblobs and clear it
    mPrevBlobs=mBlobs;
    mBlobs.clear();
    newBlobID=0;
    
    //cycle through map
    for(int i=0; i<mMapPrevToCurKeypoints.size(); i++)
    
    {
        //find value at location i of map
       int ind=mMapPrevToCurKeypoints[i];
        
        //if value is -1, create new blob
        if(ind==-1)
        {
            newBlobID++;
            mBlobs.push_back(Blob(mKeyPoints[i], newBlobID));
        }
        
        //if not, uppdate blob keypoints
        else
        {
            mPrevBlobs[ind].update( mKeyPoints[i]);
            mBlobs.push_back(mPrevBlobs[mMapPrevToCurKeypoints[ind]]);

        }

    }
    

}

void BlobDetection::draw()
{
    gl::clear( Color( 0, 0, 0 ) );
    
    gl::color( 1, 1, 1, 1 );

    //draw what image we are detecting the blobs in
    if( mBackgroundSubtracted.data && mUseBackgroundSubtraction )
    {
        gl::draw( gl::Texture::create(fromOcv(mBackgroundSubtracted)) );
    }
    else if( mTexture )
    {
        gl::draw(mTexture);
    }
    
    //draw the blobs
    for(int i=0; i<mBlobs.size(); i++)
    {
        mBlobs[i].draw();
    }
}

CINDER_APP( BlobDetection, RendererGl,
           []( BlobDetection::Settings *settings ) //note: this part is to fix the display after updating OS X 1/15/18
           {
               settings->setHighDensityDisplayEnabled( true );
               settings->setTitle("Blob Tracking Example");
               settings->setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
               //               settings->setFrameRate(FRAMERATE); //set fastest framerate
               } )
