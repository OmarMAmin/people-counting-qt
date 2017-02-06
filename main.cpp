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

int pedesterian_counting(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
    int i;
    int counter = 0;
    for(i = 0; i < num; ++i){
        int class1 = max_index(probs[i], classes);
        float prob = probs[i][class1];
        if(class1 != 0)
            continue;
        if(prob > thresh){
            counter++;
//            int width = im.h * .012;

//            if(0){
//                width = pow(prob, 1./2.)*10+1;
//                alphabet = 0;
//            }

//            printf("%s: %.0f%%\n", names[class1], prob*100);
//            int offset = class1*123457 % classes;

//            float red = get_color(2,offset,classes);
//            float green = get_color(1,offset,classes);
//            float blue = get_color(0,offset,classes);
//            float rgb[3];

//            //width = prob*20+2;

//            rgb[0] = red;
//            rgb[1] = green;
//            rgb[2] = blue;
//            box b = boxes[i];

//            int left  = (b.x-b.w/2.)*im.w;
//            int right = (b.x+b.w/2.)*im.w;
//            int top   = (b.y-b.h/2.)*im.h;
//            int bot   = (b.y+b.h/2.)*im.h;

//            if(left < 0) left = 0;
//            if(right > im.w-1) right = im.w-1;
//            if(top < 0) top = 0;
//            if(bot > im.h-1) bot = im.h-1;

//            draw_box_width(im, left, top, right, bot, width, red, green, blue);
//            if (alphabet) {
//                image label = get_label(alphabet, names[class1], (im.h*.03)/10);
//                draw_label(im, top + width, left, label, rgb);
//            }
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

int extractimages(int limit)
{
    VideoCapture cap("/home/omaramin/Desktop/dark/TownCentreXVID.avi"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("edges",1);
    int cntr = 0;
    for(;cntr<limit;cntr++)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera

        std::string savingName = "/home/omaramin/Desktop/dark/Images/" + std::to_string(++cntr) + ".jpg";
        cv::imwrite(savingName, frame);
    }
}


int main(int, char**)
{
    char * cfgfile = "cfg/yolo.cfg";
    char * weightfile = "yolo.weights";
    char * datacfg = "cfg/coco.data";
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    float thresh= 0.24;
    float hier_thresh = 0.5;

    // outputvideo
    VideoWriter out_capture("out.avi", CV_FOURCC('M','J','P','G'), 30, Size(1920,1080));

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    int j;
    float nms=.4;
    for( int counter = 1;counter < 6;counter+=2)
    {
        string basestr = "Images/";
        string imgname = to_string(counter) + ".jpg";
        string filenamestr =(basestr+imgname);
        char *filename = new char[filenamestr.length() + 1];
        strcpy(filename, filenamestr.c_str());


        // reading the Mat variable
        Mat cvimage = imread(filenamestr,1);
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
        int currentcount = pedesterian_counting(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
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

        imshow("People Counting",cvimgafter);
        waitKey(1000);
        // each frame would take a second in the output video
        for(int i =0;i<30;i++)
            out_capture.write(cvimgafter);

    }

    return 0;
}
