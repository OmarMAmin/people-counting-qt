#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"'
#include <opencv2/videoio.hpp>
#include <iostream>

#include "image.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "option_list.h"

using namespace cv;
using namespace std;

image get_label(image **characters, char *string, int size);

int count_pedesterians(int num, float thresh, float **probs, int classes)
{
    int i;
    int counter = 0;
    for(i = 0; i < num; ++i){
        // finding the index of the maximum probability
        int class1 = max_index(probs[i], classes);
        // getting the value to compare with detection threshold
        float prob = probs[i][class1];
        // person is the first label
        if(class1 != 0)
            continue;
        if(prob > thresh){
            counter++;
        }
    }
    return counter;
}


image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    image out = make_image(w, h, c);
    int i, j, k, count=0;;

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                out.data[count++] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return out;
}

image Mat_to_image(Mat img)
{
    cvtColor(img,img,CV_BGR2RGB);
    IplImage* iplimg = new IplImage(img);
    image im = ipl_to_image(iplimg);
    return im;
}

//Mat image_to_Mat(image im)
//{
//    IplImage* iplimgafter = image_to_Ipl(im,im.w,im.h,IPL_DEPTH_8U,im.c,iplimg->widthStep);
//    cv::Mat cvimgafter = cv::cvarrToMat(iplimgafter);
//    return cvimgafter;
//}

IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step)
{
   int i, j, k, count= 0;
   IplImage* src= cvCreateImage(cvSize(w, h), depth, c);

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
        src->imageData[i*step + j*c + k] = img.data[count++] * 255.;
        }
         }
          }
   cvCvtColor(src, src, CV_RGB2BGR);
   return src;
}



void pedesterian_counting_demo(string inputvideo,string outputvideo, int framescount)
{
    char * cfgfile = "cfg/yolo.cfg";
    char * weightfile = "yolo.weights";
    char * datacfg = "cfg/coco.data";
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    //char **names = get_labels(name_list);
    float thresh= 0.24;
    float hier_thresh = 0.5;


    // reading input video
    VideoCapture cap(inputvideo);

    Mat temp;
    cap>>temp;


    // outputvideo
    VideoWriter out_capture(outputvideo, CV_FOURCC('M','J','P','G'), 30, Size(temp.cols/2,temp.rows/2));

    //image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    int j;
    float nms=.4;




//    for( int counter = 1;counter < 6;counter+=2)
//    {
    for(int i = 0; i< framescount;i++)
    {
        string basestr = "Frame#";
        string imgname = to_string(i) + ":";
        string filenamestr =(basestr+imgname);
        char *filename = new char[filenamestr.length() + 1];
        strcpy(filename, filenamestr.c_str());


        // reading the Mat variable
        Mat cvimage;// = imread(filenamestr,1);
        cap>>cvimage;

        cvtColor(cvimage,cvimage,CV_BGR2RGB);
        IplImage* iplimg = new IplImage(cvimage);
        image im = ipl_to_image(iplimg);


        image sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n-1];

        box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
        if (l.softmax_tree && nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        int currentcount = count_pedesterians(l.w*l.h*l.n, thresh, probs, l.classes);
        IplImage* iplimgafter = image_to_Ipl(im,im.w,im.h,IPL_DEPTH_8U,im.c,iplimg->widthStep);
        cv::Mat cvimgafter = cv::cvarrToMat(iplimgafter);
        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);


        // labeling :
        Point upperleft(cvimgafter.cols/2-150,0);
        Point bottomright(cvimgafter.cols/2+150,80);
        string counterstr = "People Count: "+ to_string(currentcount);
        rectangle(cvimgafter,upperleft,bottomright,Scalar(0,0,255),-1);
        Point textTL(cvimgafter.cols/2-130,30);
        putText(cvimgafter, counterstr, textTL,
            FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(255,255,255), 1, CV_AA);


        // each frame would take a second in the output video
        resize(cvimgafter,cvimgafter,Size(cvimgafter.cols/2,cvimgafter.rows/2));
        imshow("People Counting",cvimgafter);
        waitKey(1000);
        for(int i =0;i<30;i++)
            out_capture.write(cvimgafter);

    }

}


int main(int argc, char**argv)
{
    string inputvideo = "../TownCentreXVID.avi";
    string outputvideo = "../output_sample.avi";
    int framescount = 10;
    if(argc >= 2)
        inputvideo = argv[1];
    if(argc >= 3)
        outputvideo = argv[2];
    if(argc >= 4)
        framescount = atoi(argv[3]);

    pedesterian_counting_demo(inputvideo,outputvideo,framescount);
    return 0;
}
