#ifndef _FAST_BLUR_H_
#define _FAST_BLUR_H_


#include "typedef.h"

//@brief  medianblur  
//@param  sptr      ptr to sptr image
//@param  dptr      ptr to dptr image
//@param  w         sptr/dptr width
//@param  h         sptr/dptr height
//@param  cn        sptr/dptr channel
//@param  sstep     sptr image bits per row
//@param  dstep     dptr image bits per row
//@param  ksize     kernel size, must be odd 
//@note   [1] no copyMakeBorder 
//        [2] only support unsigned char 
void medianBlur(const uchar*sptr, uchar* dptr, int w, int h, int cn, int sstep, int dstep, int ksize);

//TODO
void normalizedBoxBlur();
//TODO
void gaussianBlur();
//TODO
void bilateralBlur();

#endif // _FAST_BLUR_H_