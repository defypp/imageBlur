#ifndef __TYPEDEF_H__
#define __TYPEDEF_H__

//basic define  TODO
typedef unsigned char uchar;


/*  type declarations */
#ifdef __MSC_VER
#else
#include <stdint.h>
#endif

/* Compiler peculiarities */
#ifdef __GNUC__
#  define ALG_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
#  define ALG_DECL_ALIGNED(x) __declspec(align(x))
#else
#  define ALG_DECL_ALIGNED(x)
#endif


#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif


/*cpu SIMD */ 
#ifndef CV_SSE2
#define CV_SSE2  1 //&& check cpu support
#endif

/*gpu CUDA TODO*/


#endif // _TYPEDEF_H_