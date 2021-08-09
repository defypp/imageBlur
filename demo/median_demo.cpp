

#include <opencv2/opencv.hpp>

#include "fast_blur.hpp"

//compare with opencv
using namespace cv;

template<typename Func>
double kcvBench_(const Func& func, unsigned int iterNum = 1)
{
	auto t1 = cvGetTickCount();
	while (iterNum--) func();
	auto t2 = cvGetTickCount();
	return (t2 - t1) / (1000.0 * cvGetTickFrequency());
}


void demoDebug(){
	cv::Mat sptr = cv::imread("/home/robot/Documents/proj/cpp/Fastfilter/data/tree-736885.jpg",0);
	//   [0,1,2,3,4,
	//    5,6,7,8,9,
	//	  10,11,12,13,14,
	//	  15,16,17,18,19,
	//    20,21,22,23,24]
	// cv::Mat sptr(5,5,CV_8UC1);

	// sptr.at<uchar>(0,0) = 0;
	// sptr.at<uchar>(0,1) = 1;
	// sptr.at<uchar>(0,2) = 2;
	// sptr.at<uchar>(0,3) = 3;
	// sptr.at<uchar>(0,4) = 4;
	// sptr.at<uchar>(1,0) = 5;
	// sptr.at<uchar>(1,1) = 6;
	// sptr.at<uchar>(1,2) = 7;
	// sptr.at<uchar>(1,3) = 8;
	// sptr.at<uchar>(1,4) = 9;
	// sptr.at<uchar>(2,0) = 10;
	// sptr.at<uchar>(2,1) = 11;
	// sptr.at<uchar>(2,2) = 12;
	// sptr.at<uchar>(2,3) = 13;
	// sptr.at<uchar>(2,4) = 14;
	// sptr.at<uchar>(3,0) = 15;
	// sptr.at<uchar>(3,1) = 16;
	// sptr.at<uchar>(3,2) = 17;
	// sptr.at<uchar>(3,3) = 18;
	// sptr.at<uchar>(3,4) = 19;
	// sptr.at<uchar>(4,0) = 20;
	// sptr.at<uchar>(4,1) = 21;
	// sptr.at<uchar>(4,2) = 22;
	// sptr.at<uchar>(4,3) = 23;
	// sptr.at<uchar>(4,4) = 24;	

    cv::Mat dptr(sptr.rows, sptr.cols, sptr.type());

	int w = sptr.cols;
	int h = sptr.rows;
	int cn = sptr.channels();
	int step = sptr.step;
	//int srep = dptr.step;
	const uchar* ssptr = (uchar*)sptr.data;
	uchar* ddptr = (uchar*)dptr.data;
    medianBlur(ssptr, ddptr, w, h, cn, step, step,25);
	
	waitKey(0);
}

void createExceptionsImg() {

	//边界
	//边界 + ksize = 0
}

//note: 
//[1] stable test[X]
//[2] time test
//[3] limit test
void demoTest() {
	//read image from VOC dataset
	String imgFolder = "/media/robot/dev/Voc2012/voc2012/VOC2012/JPEGImages";
	std::vector<String>results;
	cv::glob(imgFolder,results);
	int numImgs = results.size();
	if(numImgs <= 0) {
		printf("no image int folder! \n");
		return;
	}
	printf("processing TIME TEST with total %d img.\n", numImgs);
	int64_t opencvTimeAccu = 0;
	int64_t demoTtimeAccu = 0;
	srand(0);
	for(int i = 0; i < numImgs; ++i) {
		if(i % 100 == 0)
			printf("processing %d img with time opencv = {%f}, demo = {%f}\n", 
			i,opencvTimeAccu / (1000.0 * cvGetTickFrequency()*(i?i:1)), 
			   demoTtimeAccu / (1000.0 * cvGetTickFrequency()*(i?i:1)));
		cv::Mat img = cv::imread(results[i], 0);
		cv::Mat dst(img.rows, img.cols, img.type());
		int v = rand() % img.rows;
		//printf("v = %d \n",v); 
		int ksize = v % 2 == 0 ? v + 1: v;
		int64_t t1 = cvGetTickCount();
		cv::medianBlur(img, dst,ksize);
		int64_t t2 = cvGetTickCount();
		opencvTimeAccu += (t2 - t1);
		int w = img.cols;
		int h = img.rows;
		int cn = img.channels();
		int step = img.step;
		const uchar* ssptr = (uchar*)img.data;
		uchar* ddptr = (uchar*)dst.data;

		t1 = cvGetTickCount();
    	medianBlur(ssptr, ddptr, w, h, cn, step, step, ksize);
		t2 = cvGetTickCount();
		demoTtimeAccu += (t2 - t1);
	}
	printf("TOTAL opencv time %f, demo time %f\n", opencvTimeAccu / (1000.0 * cvGetTickFrequency()), demoTtimeAccu / (1000.0 * cvGetTickFrequency()) );
	printf("processing STABLE TEST with exceptions.\n");

	//
}

//测试 DEMO
// TODO 边界情况测试

int main()
{
#define TEST
#ifdef TEST
	//read image from
	demoTest();
#endif// TEST

	return 0;
}
