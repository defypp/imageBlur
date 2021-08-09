//system
#include <assert.h>
#include <emmintrin.h>
#include <memory.h>
#include <stdio.h>
#include <vector>
using namespace std;
//local
#include "fast_blur.hpp"

// _Tp字节对齐
template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

const uchar g_Saturate8u[] =
{
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
	16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
	32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
	64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
	80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
	96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
	112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
	128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
	144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
	160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
	176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
	192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
	208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
	224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
	240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255
};
//配合swap a<b 进行交换 或min max使用，实现对uint的快速运算
//当t<0情况下，返回值为0, t>0返回
//swap只有三次“+”操作，一般情况需要判断，申请额外变量和赋值
#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), g_Saturate8u[(t)+256]) // release 模式下，assert不一定起作用，所以没有任何计算，可以提高速度

#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

struct MinMax8u
{
	typedef uchar value_type;
	typedef int arg_type;
	enum { SIZE = 1 };
	arg_type load(const uchar* ptr) { return *ptr; }
	void store(uchar* ptr, arg_type val) { *ptr = (uchar)val; }
	void operator()(arg_type& a, arg_type& b) const
	{
		int t = CV_FAST_CAST_8U(a - b);
		b += t; a -= t;
	}
};

struct MinMaxVec8u
{
	typedef uchar value_type;
	typedef __m128i arg_type;
	
	enum { SIZE = 16 };//寄存器总共128位，则 128 / 8 =16， 即每次可以处理16个数

	//使用_mm_loadu_si128(加载一个128位有符号整形到寄存器中，无16字节对齐，一次性取出16个数)
	arg_type load(const uchar* ptr) { return _mm_loadu_si128((const __m128i*)ptr); }
	void store(uchar* ptr, arg_type val) { _mm_storeu_si128((__m128i*)ptr, val); }
	void operator()(arg_type& a, arg_type& b) const
	{
		arg_type t = a;
		a = _mm_min_epu8(a, b);//计算最小
		b = _mm_max_epu8(b, t);//计算最大
	}
};


//simd support 
static void medianBlurSortNet(const uchar* sptr, uchar* dptr, int w, int h, int cn, int sstep, int dstep, int ksize) {

    sstep /= sizeof(sptr[0]);//某行存了多少数
	typedef MinMax8u Op;
	typedef Op::value_type T;//数据类型
	typedef Op::arg_type WT;

	typedef MinMaxVec8u VecOp;
	typedef VecOp::arg_type VT;

	int i, j, k;
	Op op;
	VecOp vop;

    bool useSIMD = true;//check cpu support sse2

	if(ksize == 3) {
        // TODO
	    // 为什么不能统一到下面的计算中？
        if (w == 1 || h == 1)
        {
            int len = w + h - 1;
            int sdelta = h == 1 ? cn : sstep;//如果是一行的情况，[1,2,3] [4,5,6] [7,8,9] sdelta=3,当ptr指向4, 则 p0 = 1, p1 = 4, p2 = 7
                                            //多行情况下， 
            int sdelta0 = h == 1 ? 0 : sstep - cn;//无论三通道还是单通道好像都是0,用于字节对齐的，见readme.md
            int ddelta = h == 1 ? cn : dstep;//同sdelta

            for (i = 0; i < len; i++, sptr += sdelta0, dptr += ddelta){
                for (j = 0; j < cn; j++, sptr++)
                {
                    WT p0 = sptr[i > 0 ? -sdelta : 0];//防止越界的处理方法
                    WT p1 = sptr[0];
                    WT p2 = sptr[i < len - 1 ? sdelta : 0];//防止越界的处理方法

                    op(p0, p1); op(p1, p2); op(p0, p1);
                    dptr[j] = (T)p1;
                }
            }
            return;
        }

        w *= cn;
        //是不是也可以放到一起
        for (i = 0; i < h; i++, dptr += dstep)
        {
            //三行的指针
            const T* row0 = sptr + MAX(i - 1, 0)*sstep;			//越界
            const T* row1 = sptr + i*sstep;
            const T* row2 = sptr + MIN(i + 1, h - 1)*sstep;		//越界
            int limit = cn;

            for (j = 0;;)
            {	
                //对channel通道的每个数
                for (; j < limit; j++)
                {
                    int j0 = j >= cn ? j - cn : j;    // 越界
                    int j2 = j < w - cn ? j + cn : j; // 越界
                    WT p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
                    WT p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
                    WT p6 = row2[j0], p7 = row2[j], p8 = row2[j2];

                    //行
                    op(p1, p2); op(p4, p5); op(p7, p8); op(p0, p1);
                    op(p3, p4); op(p6, p7); op(p1, p2); op(p4, p5);
                    op(p7, p8); 
                    //列
                    op(p0, p3); op(p5, p8); op(p4, p7);
                    op(p3, p6); op(p1, p4); op(p2, p5); op(p4, p7);
                    //对角
                    op(p4, p2); op(p6, p4); op(p4, p2);
                    dptr[j] = (T)p4;
                }

                if (limit == w)// w == 1   作为跳出的判断条件
                    break;

                //对于每行像素数量小于19(cn=3情况) j=3 < w-16-3 w> 22 (至少8个坐标)
                //寄存器是128位，16个字节
                int v = VecOp::SIZE - cn;//调用单位16个字节
                for (; j <= w - v; j += VecOp::SIZE)
                {
                    VT p0 = vop.load(row0 + j - cn), p1 = vop.load(row0 + j), p2 = vop.load(row0 + j + cn);//一次性加载16个字节
                    VT p3 = vop.load(row1 + j - cn), p4 = vop.load(row1 + j), p5 = vop.load(row1 + j + cn);
                    VT p6 = vop.load(row2 + j - cn), p7 = vop.load(row2 + j), p8 = vop.load(row2 + j + cn);

                    vop(p1, p2); vop(p4, p5); vop(p7, p8); vop(p0, p1);
                    vop(p3, p4); vop(p6, p7); vop(p1, p2); vop(p4, p5);
                    vop(p7, p8); vop(p0, p3); vop(p5, p8); vop(p4, p7);
                    vop(p3, p6); vop(p1, p4); vop(p2, p5); vop(p4, p7);
                    vop(p4, p2); vop(p6, p4); vop(p4, p2);
                    vop.store(dptr + j, p4);
                }
                limit = w;
            }
        }
    }
    else if (ksize == 5) {
        if( w == 1 || h == 1 ){
            int len = w + h - 1;
            int sdelta = h == 1 ? cn : sstep;
            int sdelta0 = h == 1 ? 0 : sstep - cn;
            int ddelta = h == 1 ? cn : dstep;

            for( i = 0; i < len; i++, sptr += sdelta0, dptr += ddelta )
                for( j = 0; j < cn; j++, sptr++ )
                {
                    int i1 = i > 0 ? -sdelta : 0;
                    int i0 = i > 1 ? -sdelta*2 : i1;
                    int i3 = i < len-1 ? sdelta : 0;
                    int i4 = i < len-2 ? sdelta*2 : i3;
                    WT p0 = sptr[i0], p1 = sptr[i1], p2 = sptr[0], p3 = sptr[i3], p4 = sptr[i4];

                    op(p0, p1); op(p3, p4); op(p2, p3); op(p3, p4); op(p0, p2);
                    op(p2, p4); op(p1, p3); op(p1, p2);
                    dptr[j] = (T)p2;
                }
            return;
        }
        w *= cn;
        for( i = 0; i < h; i++, dptr += dstep ){
            const T* row[5]; // 5 line ptr
            row[0] = sptr + MAX(i - 2, 0)*sstep;
            row[1] = sptr + MAX(i - 1, 0)*sstep;
            row[2] = sptr + i*sstep;
            row[3] = sptr + MIN(i + 1, h-1)*sstep;
            row[4] = sptr + MIN(i + 2, h-1)*sstep;
            //
            int limit = useSIMD ? cn*2 : w;// 批处理分界线
            // 对前后2×cn的像素进行特殊处理
            for(j = 0;; ){
                for( ; j < limit; j++ ){
                    WT p[25];
                    int j1 = j >= cn ? j - cn : j; //limits
                    int j0 = j >= cn*2 ? j - cn*2 : j1;
                    int j3 = j < w - cn ? j + cn : j;
                    int j4 = j < w - cn*2 ? j + cn*2 : j3;
                    for( k = 0; k < 5; k++ )
                    {
                        const T* rowk = row[k];
                        p[k*5] = rowk[j0]; p[k*5+1] = rowk[j1];
                        p[k*5+2] = rowk[j]; p[k*5+3] = rowk[j3];
                        p[k*5+4] = rowk[j4];
                    }
                    //5×5的kernel比较
                    op(p[1], p[2]); op(p[0], p[1]); op(p[1], p[2]); 
                    op(p[4], p[5]); op(p[3], p[4]);op(p[4], p[5]); 
                    op(p[0], p[3]); op(p[2], p[5]); op(p[2], p[3]); 
                    op(p[1], p[4]);op(p[1], p[2]); op(p[3], p[4]); 

                    op(p[7], p[8]); op(p[6], p[7]); op(p[7], p[8]);
                    op(p[10], p[11]); op(p[9], p[10]); op(p[10], p[11]); 
                    op(p[6], p[9]); op(p[8], p[11]);op(p[8], p[9]); 
                    op(p[7], p[10]); op(p[7], p[8]); op(p[9], p[10]); 
                    
                    op(p[0], p[6]);op(p[4], p[10]); op(p[4], p[6]); op(p[2], p[8]); op(p[2], p[4]); op(p[6], p[8]);
                    op(p[1], p[7]); op(p[5], p[11]); op(p[5], p[7]); op(p[3], p[9]); op(p[3], p[5]);
                    op(p[7], p[9]); op(p[1], p[2]); op(p[3], p[4]); op(p[5], p[6]); op(p[7], p[8]);
                    op(p[9], p[10]); op(p[13], p[14]); op(p[12], p[13]); op(p[13], p[14]); op(p[16], p[17]);
                    op(p[15], p[16]); op(p[16], p[17]); op(p[12], p[15]); op(p[14], p[17]); op(p[14], p[15]);
                    op(p[13], p[16]); op(p[13], p[14]); op(p[15], p[16]); op(p[19], p[20]); op(p[18], p[19]);
                    op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[21], p[23]); op(p[22], p[24]);
                    op(p[22], p[23]); op(p[18], p[21]); op(p[20], p[23]); op(p[20], p[21]); op(p[19], p[22]);
                    op(p[22], p[24]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[12], p[18]);
                    op(p[16], p[22]); op(p[16], p[18]); op(p[14], p[20]); op(p[20], p[24]); op(p[14], p[16]);
                    op(p[18], p[20]); op(p[22], p[24]); op(p[13], p[19]); op(p[17], p[23]); op(p[17], p[19]);
                    op(p[15], p[21]); op(p[15], p[17]); op(p[19], p[21]); op(p[13], p[14]); op(p[15], p[16]);
                    op(p[17], p[18]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[0], p[12]);
                    op(p[8], p[20]); op(p[8], p[12]); op(p[4], p[16]); op(p[16], p[24]); op(p[12], p[16]);
                    op(p[2], p[14]); op(p[10], p[22]); op(p[10], p[14]); op(p[6], p[18]); op(p[6], p[10]);
                    op(p[10], p[12]); op(p[1], p[13]); op(p[9], p[21]); op(p[9], p[13]); op(p[5], p[17]);
                    op(p[13], p[17]); op(p[3], p[15]); op(p[11], p[23]); op(p[11], p[15]); op(p[7], p[19]);
                    op(p[7], p[11]); op(p[11], p[13]); op(p[11], p[12]);
                    dptr[j] = (T)p[12];
                }

                if( limit == w )
                    break;

                for( ; j <= w - VecOp::SIZE - cn*2; j += VecOp::SIZE ){
                    VT p[25];
                    for( k = 0; k < 5; k++ ){
                        const T* rowk = row[k];
                        p[k*5] = vop.load(rowk+j-cn*2); p[k*5+1] = vop.load(rowk+j-cn);
                        p[k*5+2] = vop.load(rowk+j); p[k*5+3] = vop.load(rowk+j+cn);
                        p[k*5+4] = vop.load(rowk+j+cn*2);
                    }

                    vop(p[1], p[2]); vop(p[0], p[1]); vop(p[1], p[2]); vop(p[4], p[5]); vop(p[3], p[4]);
                    vop(p[4], p[5]); vop(p[0], p[3]); vop(p[2], p[5]); vop(p[2], p[3]); vop(p[1], p[4]);
                    vop(p[1], p[2]); vop(p[3], p[4]); vop(p[7], p[8]); vop(p[6], p[7]); vop(p[7], p[8]);
                    vop(p[10], p[11]); vop(p[9], p[10]); vop(p[10], p[11]); vop(p[6], p[9]); vop(p[8], p[11]);
                    vop(p[8], p[9]); vop(p[7], p[10]); vop(p[7], p[8]); vop(p[9], p[10]); vop(p[0], p[6]);
                    vop(p[4], p[10]); vop(p[4], p[6]); vop(p[2], p[8]); vop(p[2], p[4]); vop(p[6], p[8]);
                    vop(p[1], p[7]); vop(p[5], p[11]); vop(p[5], p[7]); vop(p[3], p[9]); vop(p[3], p[5]);
                    vop(p[7], p[9]); vop(p[1], p[2]); vop(p[3], p[4]); vop(p[5], p[6]); vop(p[7], p[8]);
                    vop(p[9], p[10]); vop(p[13], p[14]); vop(p[12], p[13]); vop(p[13], p[14]); vop(p[16], p[17]);
                    vop(p[15], p[16]); vop(p[16], p[17]); vop(p[12], p[15]); vop(p[14], p[17]); vop(p[14], p[15]);
                    vop(p[13], p[16]); vop(p[13], p[14]); vop(p[15], p[16]); vop(p[19], p[20]); vop(p[18], p[19]);
                    vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[21], p[23]); vop(p[22], p[24]);
                    vop(p[22], p[23]); vop(p[18], p[21]); vop(p[20], p[23]); vop(p[20], p[21]); vop(p[19], p[22]);
                    vop(p[22], p[24]); vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[12], p[18]);
                    vop(p[16], p[22]); vop(p[16], p[18]); vop(p[14], p[20]); vop(p[20], p[24]); vop(p[14], p[16]);
                    vop(p[18], p[20]); vop(p[22], p[24]); vop(p[13], p[19]); vop(p[17], p[23]); vop(p[17], p[19]);
                    vop(p[15], p[21]); vop(p[15], p[17]); vop(p[19], p[21]); vop(p[13], p[14]); vop(p[15], p[16]);
                    vop(p[17], p[18]); vop(p[19], p[20]); vop(p[21], p[22]); vop(p[23], p[24]); vop(p[0], p[12]);
                    vop(p[8], p[20]); vop(p[8], p[12]); vop(p[4], p[16]); vop(p[16], p[24]); vop(p[12], p[16]);
                    vop(p[2], p[14]); vop(p[10], p[22]); vop(p[10], p[14]); vop(p[6], p[18]); vop(p[6], p[10]);
                    vop(p[10], p[12]); vop(p[1], p[13]); vop(p[9], p[21]); vop(p[9], p[13]); vop(p[5], p[17]);
                    vop(p[13], p[17]); vop(p[3], p[15]); vop(p[11], p[23]); vop(p[11], p[15]); vop(p[7], p[19]);
                    vop(p[7], p[11]); vop(p[11], p[13]); vop(p[11], p[12]);
                    vop.store(dptr+j, p[12]);
                }

                limit = w; 
            } 
        }      
    }

}

static void medianBlur8uOm(const uchar* sptr, uchar* dptr, int w, int h, int cn, int sstep, int dstep, int ksize)
{
    #define N  16
    int     zone0[4][N];//zone0 进行第一次快速查找 如果 考虑
    int     zone1[4][N*N];

       int     n2 = ksize*ksize/2;
    const uchar*  sptrMax = sptr + h * sstep;
    const uchar*  sptrTmp = sptr;

    //push pix value in zone0 and zone1 
    #define UPDATE_ACC01( pix, cn, op ) \
    {                                   \
        int p = (pix);                  \
        zone1[cn][p] op;                \
        zone0[cn][p >> 4] op;           \
    }
    // TODO
    int x, y;
    int m = ksize;
    // init accumulator
        // memset( zone0, 0, sizeof(zone0[0])*cn );//根据cn初始化直方图
        // memset( zone1, 0, sizeof(zone1[0])*cn );//已经重新初始化为什么需要反向再来一遍  结构体维持
    //列遍历
    for( x = 0; x < w; x++, sptr += cn, dptr += cn ){
        uchar* dst_cur = dptr;
        const uchar* src_top = sptr;
        const uchar* src_bottom = sptr;
        int k, c;
        int src_step1 = sstep;
        int dst_step1 = dstep;

        if( x % 2 != 0 )// 反过来是因为需要使用上一次的计算
        {
            src_bottom = src_top += sstep*(h-1);//指向最后一行
            dst_cur += dstep*(h-1);             //指向最后一行
            src_step1 = -src_step1;            //倒着计算的意思
            dst_step1 = -dst_step1;            //
        }

        // init accumulator
        memset( zone0, 0, sizeof(zone0[0])*cn );//根据cn初始化直方图
        memset( zone1, 0, sizeof(zone1[0])*cn );//已经重新初始化为什么需要反向再来一遍  结构体维持
        // ？why only half kernel
        // 动态边界的调整
        for( y = 0; y <= m/2; y++ )
        {
            for( c = 0; c < cn; c++ )
            {
                if( y > 0 )
                {
                    for( k = 0; k < m*cn; k += cn )
                        UPDATE_ACC01( src_bottom[k+c], c, ++ );
                }
                else
                {
                    for( k = 0; k < m*cn; k += cn ){
                        UPDATE_ACC01(src_bottom[k+c], c, += m/2+1 ); // 为什么y=0的情况下需要+ m/2  右侧越界全为0
                        //图像扩展边界的过程中使用了BORDER_REPLICATE，当y=0,需要+ m/2+1，从而得到m*m个数取中值
                        //src_bottom越界如何处理？
                        //(x * cn + k + c) > step ? 0 : 
                    }
                }
            }

            if( (src_step1 > 0 && y < h-1) ||
                (src_step1 < 0 && h-y-1 > 0) )
                src_bottom += src_step1;
        }

        for( y = 0; y < h; y++, dst_cur += dst_step1 )
        {
            // find median
            // zone0 zone1 查找中值
            // 处理[x=0,y=0] 处的中值
            for( c = 0; c < cn; c++ )
            {
                int s = 0;
                for( k = 0; ; k++ )
                {
                    int t = s + zone0[c][k];
                    if( t > n2 ) break;//当出现的像素次数累加>滑动窗口一半大小跳出，表示在zone0的第k个区间存在中值
                    s = t;
                }

                for( k *= N; ;k++ )
                {
                    s += zone1[c][k];
                    if( s > n2 ) break;//在zone1的第k个区间内查找具体的中值
                }

                dst_cur[c] = (uchar)k;//k [0-256] bins
            }

            if( y+1 == h) //防止src_bottom越界访问
                break;
            // what for？
            if( cn == 1 )
            {   
                //将src_top的值对zone的影响消除，使用src_bottom重新加入
                //此时src_top指向x=a行,src_bottom指向a+ m/2 + 1 行
                for( k = 0; k < m; k++ )
                {
                    int p = src_top[k];
                    int q = src_bottom[k];
                    zone1[0][p]--;
                    zone0[0][p>>4]--;
                    zone1[0][q]++;
                    zone0[0][q>>4]++;
                }
            }
            else if( cn == 3 )
            {
                for( k = 0; k < m*3; k += 3 )
                {
                    UPDATE_ACC01( src_top[k], 0, -- );
                    UPDATE_ACC01( src_top[k+1], 1, -- );
                    UPDATE_ACC01( src_top[k+2], 2, -- );

                    UPDATE_ACC01( src_bottom[k], 0, ++ );
                    UPDATE_ACC01( src_bottom[k+1], 1, ++ );
                    UPDATE_ACC01( src_bottom[k+2], 2, ++ );
                }
            }
            else
            {
                assert( cn == 4 );
                for( k = 0; k < m*4; k += 4 )
                {
                    UPDATE_ACC01( src_top[k], 0, -- );
                    UPDATE_ACC01( src_top[k+1], 1, -- );
                    UPDATE_ACC01( src_top[k+2], 2, -- );
                    UPDATE_ACC01( src_top[k+3], 3, -- );

                    UPDATE_ACC01( src_bottom[k], 0, ++ );
                    UPDATE_ACC01( src_bottom[k+1], 1, ++ );
                    UPDATE_ACC01( src_bottom[k+2], 2, ++ );
                    UPDATE_ACC01( src_bottom[k+3], 3, ++ );
                }
            }

            if( (src_step1 > 0 && src_bottom + src_step1 < sptrMax) ||
                (src_step1 < 0 && src_bottom + src_step1 >= sptr) )
                src_bottom += src_step1;

            if( y >= m/2 )
                src_top += src_step1;
        }
        // adjust cols TODO
    }
}

//****************************************medianBlur8uO1*******************************************
typedef ushort HT;

typedef struct
{
    HT coarse[16];
    HT fine[16][16];
} Histogram;

#if CV_SSE2
#define MEDIAN_HAVE_SIMD 1

static inline void histogram_add_simd( const HT x[16], HT y[16] )
{
    const __m128i* rx = (const __m128i*)x;//将x转换为16个字节为单位的指针，则rx+0表示第一个16字节，rx+1表示第二个16字节
    __m128i* ry = (__m128i*)y;
    __m128i r0 = _mm_add_epi16(_mm_load_si128(ry+0),_mm_load_si128(rx+0));//注意溢出情况 add/adds
    __m128i r1 = _mm_add_epi16(_mm_load_si128(ry+1),_mm_load_si128(rx+1));
    _mm_store_si128(ry+0, r0);
    _mm_store_si128(ry+1, r1);
}

static inline void histogram_sub_simd( const HT x[16], HT y[16] )
{
    const __m128i* rx = (const __m128i*)x;
    __m128i* ry = (__m128i*)y;
    __m128i r0 = _mm_sub_epi16(_mm_load_si128(ry+0),_mm_load_si128(rx+0));
    __m128i r1 = _mm_sub_epi16(_mm_load_si128(ry+1),_mm_load_si128(rx+1));
    _mm_store_si128(ry+0, r0);
    _mm_store_si128(ry+1, r1);
}

#else
#define MEDIAN_HAVE_SIMD 0
#endif

static inline void histogram_add( const HT x[16], HT y[16] )
{
    int i;
    for( i = 0; i < 16; ++i )
        y[i] = (HT)(y[i] + x[i]);
}

static inline void histogram_sub( const HT x[16], HT y[16] )
{
    int i;
    for( i = 0; i < 16; ++i )
        y[i] = (HT)(y[i] - x[i]);
}

static inline void histogram_muladd( int a, const HT x[16], HT y[16] )
{
    for( int i = 0; i < 16; ++i )
        y[i] = (HT)(y[i] + a * x[i]);
}

// h_coarse cn=3 n r 的内存分布
// 第一行  cn=0 {[0-15]_col0 [0-15]_col1...[0-15]_coln} cn=1{[0-15]_col0 [0-15]_col1...[0-15]_coln}  cn=2{[0-15]_col0 [0-15]_col1...[0-15]_coln} 
//...同上

static void medianBlur8uO1(const uchar* sptr, uchar* dptr, int w, int h, int cn, int sstep, int dstep, int ksize) {
/**
 * HOP is short for Histogram OPeration. This macro makes an operation \a op on
 * histogram \a h for pixel value \a x. It takes care of handling both levels.
 */

#define HOP(h,x,op) \
    h.coarse[x>>4] op, \
    *((HT*)h.fine + x) op

// 16 * (n*(16*c+(x>>4)) + j) + (x & 0xF) = 16×16×*n*c + 16（n*x>>4+j)  + (x & 0xF)  = 混乱！！！！！！！！！！！！
//   *** n=512,表示一个大块的列
// coarse 1 * 16 * (STRIPE_SIZE + 2*r) * cn + 16; 列存储
// fine 行存储！
#define COP(c,j,x,op) \
    h_coarse[ 16*(n*c+j) + (x>>4) ] op, \
    h_fine[ 16 * (n*(16*c+(x>>4)) + j) + (x & 0xF) ] op
    //看不懂h_fine的更新！！！

    //int cn = _dst.channels(), 
    int m = h;
    int r = (ksize-1)/2;

    Histogram ALG_DECL_ALIGNED(16) H[4]; // 记录kernel的滤波直方图信息
    HT ALG_DECL_ALIGNED(16) luc[4][16];  // 

    int STRIPE_SIZE = std::min( w, 512/cn );// 列分块的原理？ 512会不会根256统计有关系

    // 结构体 好难 理解
    vector<HT> _h_coarse(1 * 16 * (STRIPE_SIZE + 2*r) * cn + 16);// 记录colum的直方图统计信息
    vector<HT> _h_fine(16 * 16 * (STRIPE_SIZE + 2*r) * cn + 16);
    HT* h_coarse = alignPtr(&_h_coarse[0], 16);//指针的字节对齐
    HT* h_fine = alignPtr(&_h_fine[0], 16);//
#if MEDIAN_HAVE_SIMD
    bool useSIMD = true;
#endif
    // col -> row
    for( int x = 0; x < w; x += STRIPE_SIZE ) //STRIPE_SIZE列进行一次处理
    {
        int i, j, k, c, n = std::min(w - x, STRIPE_SIZE) + r*2;//处理边界
        const uchar* src = sptr + x*cn;
        uchar* dst = dptr + (x - r)*cn;

        memset( h_coarse, 0, 16*n*cn*sizeof(h_coarse[0]) );//不越界的判断
        memset( h_fine, 0, 16*16*n*cn*sizeof(h_fine[0]) );//

        // First row initialization
        // column 直方图初始化前n列 
        //h_coarse 每行按照通道数的增加依次统计像素的直方图
        for( c = 0; c < cn; c++ )
        {
            for( j = 0; j < n; j++ )//第一行n列
                COP( c, j, src[cn*j+c], += r+2 );// 填充边界

            for( i = 1; i < r; i++ )// 遍历kernel行
            {
                const uchar* p = src + sstep*std::min(i, m-1);
                for ( j = 0; j < n; j++ )
                    COP( c, j, p[cn*j+c], ++ );
            }
        }

        for( i = 0; i < m; i++ )
        {
            const uchar* p0 = src + sstep * std::max( 0, i-r-1 );
            const uchar* p1 = src + sstep * std::min( m-1, i+r );
            // kernel histogram init
            memset( H, 0, cn*sizeof(H[0]) );
            memset( luc, 0, cn*sizeof(luc[0]) );
            for( c = 0; c < cn; c++ )
            {
                // Update column histograms for the entire row.
                for( j = 0; j < n; j++ )
                {
                    COP( c, j, p0[j*cn + c], -- );
                    COP( c, j, p1[j*cn + c], ++ );
                }
                // First column initialization
                // 将h_fine直方图的信息添加到H的kernel直方图中
                for( k = 0; k < 16; ++k )
                    histogram_muladd( 2*r+1, &h_fine[16*n*(16*c+k)], &H[c].fine[k][0] ); //获取 h_fine第一列的指针 和 H的最低位指针，并对第一列的数据进行赋值
                    // for i in 16 : y[i] = (HT)(y[i] + a * x[i]);

            #if MEDIAN_HAVE_SIMD
                if( useSIMD )
                {   
                    //前2r列赋值
                    for( j = 0; j < 2*r; ++j )
                        histogram_add_simd( &h_coarse[16*(n*c+j)], H[c].coarse );
                    //依次增加一列，更新H矩阵
                    //hfine和H.fine的前2r列不需要更新吗？，只有在find median at fine level会需要
                    for( j = r; j < n-r; j++ )
                    {
                        //t = (2r+1)^2/2;
                        int t = 2*r*r + 2*r, b, sum = 0;
                        HT* segment;
                        //
                        histogram_add_simd( &h_coarse[16*(n*c + std::min(j+r,n-1))], H[c].coarse );

                        // Find median at coarse level
                        for ( k = 0; k < 16 ; ++k )
                        {
                            sum += H[c].coarse[k];
                            if ( sum > t )
                            {
                                sum -= H[c].coarse[k];
                                break;
                            }
                        }
                        assert( k < 16 );

                        /* Update corresponding histogram segment*/
                        if ( luc[c][k] <= j-r )
                        {
                            memset( &H[c].fine[k], 0, 16 * sizeof(HT) );
                            for ( luc[c][k] = j-r; luc[c][k] < MIN(j+r+1,n); ++luc[c][k] )
                                histogram_add_simd( &h_fine[16*(n*(16*c+k)+luc[c][k])], H[c].fine[k] );

                            if ( luc[c][k] < j+r+1 )
                            {
                                histogram_muladd( j+r+1 - n, &h_fine[16*(n*(16*c+k)+(n-1))], &H[c].fine[k][0] );
                                luc[c][k] = (HT)(j+r+1);
                            }
                        }
                        else
                        {
                            for ( ; luc[c][k] < j+r+1; ++luc[c][k] )
                            {
                                histogram_sub_simd( &h_fine[16*(n*(16*c+k)+MAX(luc[c][k]-2*r-1,0))], H[c].fine[k] );
                                histogram_add_simd( &h_fine[16*(n*(16*c+k)+MIN(luc[c][k],n-1))], H[c].fine[k] );
                            }
                        }

                        histogram_sub_simd( &h_coarse[16*(n*c+MAX(j-r,0))], H[c].coarse );

                        /* Find median in segment */
                        segment = H[c].fine[k];
                        for ( b = 0; b < 16 ; b++ )
                        {
                            sum += segment[b];
                            if ( sum > t )
                            {
                                dst[dstep*i+cn*j+c] = (uchar)(16*k + b);
                                break;
                            }
                        }
                        assert( b < 16 );
                    }
                }
                else
            #endif
                {
                    for( j = 0; j < 2*r; ++j )
                        histogram_add( &h_coarse[16*(n*c+j)], H[c].coarse );

                    for( j = r; j < n-r; j++ )
                    {
                        int t = 2*r*r + 2*r, b, sum = 0;
                        HT* segment;

                        histogram_add( &h_coarse[16*(n*c + std::min(j+r,n-1))], H[c].coarse );

                        // Find median at coarse level
                        for ( k = 0; k < 16 ; ++k )
                        {
                            sum += H[c].coarse[k];
                            if ( sum > t )
                            {
                                sum -= H[c].coarse[k];
                                break;
                            }
                        }
                        assert( k < 16 );

                        /* Update corresponding histogram segment */
                        if ( luc[c][k] <= j-r )
                        {
                            memset( &H[c].fine[k], 0, 16 * sizeof(HT) );
                            for ( luc[c][k] = j-r; luc[c][k] < MIN(j+r+1,n); ++luc[c][k] )
                                histogram_add( &h_fine[16*(n*(16*c+k)+luc[c][k])], H[c].fine[k] );

                            if ( luc[c][k] < j+r+1 )
                            {
                                histogram_muladd( j+r+1 - n, &h_fine[16*(n*(16*c+k)+(n-1))], &H[c].fine[k][0] );
                                luc[c][k] = (HT)(j+r+1);
                            }
                        }
                        else
                        {
                            for ( ; luc[c][k] < j+r+1; ++luc[c][k] )
                            {
                                histogram_sub( &h_fine[16*(n*(16*c+k)+MAX(luc[c][k]-2*r-1,0))], H[c].fine[k] );
                                histogram_add( &h_fine[16*(n*(16*c+k)+MIN(luc[c][k],n-1))], H[c].fine[k] );
                            }
                        }

                        histogram_sub( &h_coarse[16*(n*c+MAX(j-r,0))], H[c].coarse );

                        /* Find median in segment */
                        segment = H[c].fine[k];
                        for ( b = 0; b < 16 ; b++ )
                        {
                            sum += segment[b];
                            if ( sum > t )
                            {
                                dst[dstep*i+cn*j+c] = (uchar)(16*k + b);
                                break;
                            }
                        }
                        assert( b < 16 );
                    }
                }
            }
        }
    }

#undef HOP
#undef COP
}

void medianBlur(const uchar*sptr, uchar* dptr, int w, int h, int cn,int sstep, int dstep,int ksize){
    //check exception
    if(sptr == NULL || dptr==NULL || sptr == dptr) return;
    if(w <= 0 || h <= 0 ) return;
    if(cn != 1 && cn != 3 && cn != 4) return;
    if(ksize <= 1) {
        //cp sptr to dptr per line
        for(int i = 0; i < h;++i)
            memcpy((uchar*)sptr+ i * sstep, dptr + i* dstep, sizeof(uchar)*sstep);//this will be error if step not same
        return;
    }
    if(ksize % 2 != 1) return;
    bool useSortNet = (ksize == 3 || ksize == 5)&& int(CV_SSE2);
    
    // image preprocess
   // if(!useSortNet) 0.
        //TODO after copy border step will be 
        //cv::copyMakeBorder( src0, src, 0, 0, ksize/2, ksize/2, BORDER_REPLICATE )
    if(useSortNet) {
        medianBlurSortNet(sptr, dptr, w, h, cn, sstep, dstep, ksize);
        return;
    }
    double img_size_mp = (double)(w * h)/(1 << 20); // 1024 *1024 compare

    // 根据核的大小来采用不同的方法进行排序
    // 根据图像大小动态调整Om/01方法
    // [0-1024]-> k=15
    // [1024,4048] -> k =9
    // [4048,+]  -> k = 5 
    bool useSIMD = CV_SSE2;//check cpu support
    if( ksize <= 3 + (img_size_mp < 1 ? 12 : img_size_mp < 4 ? 6 : 2)*(useSIMD))//分类情况跟我想象的不太一样！！
        medianBlur8uOm(sptr, dptr, w, h, cn, sstep, dstep, ksize);
    else
        medianBlur8uO1(sptr, dptr, w, h, cn, sstep, dstep, ksize);
}







