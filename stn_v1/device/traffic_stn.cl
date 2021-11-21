
//stn_conv1
// Implements conv + relu + maxpooling

 __constant int  mp_stn_conv1_base[4] = {0,1,26,27} ;


__kernel void stn_conv1 (__global float * restrict  im_g,
	                     __global float * restrict   w_g ,
	                     __global float * restrict   b_g ,
       		             __global float * restrict   out_g 
		                )  

{
// Image Size = 32*32*3
// Convolved with 5*5*3 (100) filters generating 100 output feature maps
// No zero padding , stride = 1

__local float im_buffer[1024];

// 32 - 5 + 1 = 28
// 28*28 = 784

__local int  toep_buffer[676][49];
__local float  filter_patch[49][50] ; //100 filters
__local float  op_buffer[676][50];
__local int m2[169][4] ;     //For maxpooling with stride 2
__local float bias[50];
__local float out_mem[8450] ;


// Registers
int base[49];

//Initializing output buffer 

// for (int i = 0 ;i < 676 ; i++)
//      {
//      for (int j = 0 ; j < 50; j++)
//      {
//        op_buffer[i][j]  = 0; 
//      }
//      }


for (int i = 0; i < 7 ; i++) 
    {
    for (int j = 0; j < 7 ; j ++)
	 {
	  base[i*7 + j] = 32*i + j ;
	 }
	}

  for(int i = 0 ; i < 50 ; i++)
	{
	  bias[i] = b_g[i] ;
	}



//Toep Buffer contains the addresses
//Toeplitz Computation
  for (int i = 0 ; i < 26 ; i++)
   {
  	for (int j = 0 ; j < 26 ; j++)
   	{
	 for (int k = 0 ; k < 50 ; k ++)
	     {
		toep_buffer[26*i + j][k] = i*32 + base[k] + j ;
	     }
	}
   }

//m2 buffer for max pooling
 for (int i = 0 ; i < 13 ; i++)
    {

    for (int j = 0 ; j < 13 ; j++)
     {
 	for (int k = 0; k < 4 ; k++)
	    {
	      m2[13*i + j][k] = mp_stn_conv1_base[k] + 2*j + 52*i ;
	    }
    }
  }

   
  
	
	// Loading data into im_buffer from global memory channel wise.

	 for (int i = 0 ; i < 1024 ; i++)
		{
		 im_buffer[i] = im_g[i ];
		}
	
	// Creating Filter patch matrix 
	
	for (int  k = 0 ; k < 50 ; k ++)
	     {
	     for (int i = 0; i < 49 ; i++)
	     {
	        filter_patch[i][k] = w_g[49*k + i ];
	     }
	     }	
	
	// computing output buffer of convolution

	 for (int i = 0 ; i < 676 ; i ++)
	  {
	   for (int j = 0 ; j < 49 ; j++)
	   {
	    for (int k = 0 ; k < 50 ; k++)
	     {
	       op_buffer[i][k] += im_buffer[toep_buffer[i][j]]*filter_patch[j][k];
	     }
	   }

	  }
		
		
	  // ELU 
	 for (int i = 0 ; i < 676 ; i ++)
	 {
	  for(int j = 0; j < 50 ; j ++)
	  {
	   if (op_buffer[i][j] < 0)
	     {
		op_buffer[i][j] = exp(op_buffer[i][j]+bias[j])-1;
	     }
	    else
	    {
	      op_buffer[i][j] = op_buffer[i][j] + bias[j] ;
	    }
	  }
	 }

	//Maxpooling 
 	float a1 , b1, c1, d1 ;
	float max1,max2 ;
	float max3;

	for(int k = 0 ;k< 50 ; k++)
	  {
	   for (int i = 0 ; i < 169 ; i++)
	   {
	    a1 = op_buffer[m2[i][0]][k]; 
	    b1 = op_buffer[m2[i][1]][k]; 
	    c1 = op_buffer[m2[i][2]][k]; 
	    d1 = op_buffer[m2[i][3]][k]; 

	    if (a1 > b1)
	    {
	    max1 = a1;
	    }
	    else 
	    {
	    max1 = b1;
	    }
	    if (c1 > d1)
	    {
	    max2 = c1;
	    }
	    else
	    {
	    max2 = d1;
	    }
	    if (max1 > max2)
	    {
	    max3 = max1;
	    }
	    else 
	    {
	    max3 = max2 ;
	    }
	      out_mem[169*k + i] = max3;
	 
	    }
	    }
        for (int i = 0; i < 8450 ; i++)
        {
            out_g[i] = out_mem[i] ;

        }
        }
/***************************************stn_conv2.cl******************************************/
// Implements conv + relu + maxpooling


 __constant int  mp_stn_conv2_base[4] = {0,1,8,9} ;

__kernel void stn_conv2 (__global float * restrict im_g,
	             __global float *     restrict  w_g ,
	             __global float *     restrict  b_g ,
    		     __global float *     restrict  out_g 
		                )  

{
// Image Size = 32*32*3
// Convolved with 5*5*3 (100) filters generating 100 output feature maps
// No zero padding , stride = 1

__local float im_buffer[169];

// 32 - 5 + 1 = 28
// 28*28 = 784

__local int  toep_buffer[81][25];
__local float  filter_patch[25][100] ; //100 filters
__local float  op_buffer[81][100];
__local int m2[16][4] ;     //For maxpooling with stride 2
__local float bias[100];

__local float out_mem[1600];



// Registers
int base[25];

//Initializing output buffer 

// for (int i = 0 ;i < 81 ; i++)
//      {
//      for (int j = 0 ; j < 100; j++)
//      {
//        op_buffer[i][j]  = 0; 
//      }
//      }


for (int i = 0; i < 5 ; i++) 
    {
    for (int j = 0; j < 5 ; j ++)
	 {
	  base[i*5 + j] = 13*i + j ;
	 }
	}

  for(int i = 0 ; i < 100 ; i++)
	{
	  bias[i] = b_g[i] ;
	}



//Toep Buffer contains the addresses
//Toeplitz Computation
  for (int i = 0 ; i < 9 ; i++)
   {
  	for (int j = 0 ; j < 9 ; j++)
   	{
	 for (int k = 0 ; k < 25 ; k ++)
	     {
		toep_buffer[9*i + j][k] = i*13 + base[k] + j ;
	     }
	}
   }

//m2 buffer for max pooling
 for (int i = 0 ; i < 4 ; i++)
    {

    for (int j = 0 ; j < 4 ; j++)
     {
 	for (int k = 0; k < 4 ; k++)
	    {
	      m2[4*i + j][k] = mp_stn_conv2_base[k] + 2*j + 16*i ;
	    }
    }
  }

  //Channel Loop
   
  for (int c = 0 ; c < 50 ;c++)
	 {
	
	// Loading data into im_buffer from global memory channel wise.

	 for (int i = 0 ; i < 169 ; i++)
		{
		 im_buffer[i] = im_g[169*c + i ];
		}
	
	// Creating Filter patch matrix 
	
	for (int  k = 0 ; k < 100 ; k ++)
	     {
	     for (int i = 0; i < 25 ; i++)
	     {
	        filter_patch[i][k] = w_g[1250*k + 25*c + i ];
	     }
	     }	
	
	// computing output buffer of convolution

	 for (int i = 0 ; i < 81 ; i ++)
	  {
	   for (int j = 0 ; j < 25 ; j++)
	   {
	    for (int k = 0 ; k < 100 ; k++)
	     {
	       op_buffer[i][k] += im_buffer[toep_buffer[i][j]]*filter_patch[j][k];
	     }
	   }

	  }
	}	
		
	  // Leaky Relu , adding bias 
	 for (int i = 0 ; i < 81 ; i ++)
	 {
	  for(int j = 0; j < 100 ; j ++)
	  {
	   if (op_buffer[i][j] < 0)
	     {
		op_buffer[i][j] = exp(op_buffer[i][j] + bias[j] )-1;
	     }
	    else
	    {
	      op_buffer[i][j] = op_buffer[i][j] + bias[j] ;
	    }
	  }
	 }

	//Maxpooling 
 	float a1 , b1, c1, d1 ;
	float max1,max2 ;
	float max3;

	for(int k = 0 ;k< 100 ; k++)
	  {
	   for (int i = 0 ; i < 16 ; i++)
	   {
	    a1 = op_buffer[m2[i][0]][k]; 
	    b1 = op_buffer[m2[i][1]][k]; 
	    c1 = op_buffer[m2[i][2]][k]; 
	    d1 = op_buffer[m2[i][3]][k]; 

	    if (a1 > b1)
	    {
	    max1 = a1;
	    }
	    else 
	    {
	    max1 = b1;
	    }
	    if (c1 > d1)
	    {
	    max2 = c1;
	    }
	    else
	    {
	    max2 = d1;
	    }
	    if (max1 > max2)
	    {
	    max3 = max1;
	    }
	    else 
	    {
	    max3 = max2 ;
	    }
	      out_mem[16*k + i] = max3;
	    }
	    }
	  for (int i = 0; i <1600 ; i++)
          {
            out_g[i]= out_mem[i] ;

          }

	    
	}
/*******************************************************FC1.cl*********************************************************/
__constant int W_FC1_height = 100 ;
__constant int W_FC1_width  = 1600 ;
__constant int X_FC1_width = 1 ;




//__attribute__((reqd_work_group_size(1,16,1)))

__kernel void FC1 (    __global float *restrict X,
		       __global float *restrict W,
		       __global float *restrict bias,
		       __global float *restrict Y
		  )



{

const int global_x = get_global_id(0);    //globadXDx of a work item
const int global_y = get_global_id(1);    //globadXDy of a work item


float  acc = 0.0f;

for (int k=0 ; k<W_FC1_width ; k++) 
{
acc += W[global_y*W_FC1_width + k]*X[k*X_FC1_width + global_x];
}

//Store the result.
Y[global_y*X_FC1_width + global_x] =acc + bias[global_y*X_FC1_width + global_x];
}

/***********************************************FC2.cl*************************************/

__kernel void FC2(__global  float  * restrict im_g,
		          __global  float * restrict  w_g,
                  __global  float * restrict  b_g,
                  __global  float * restrict  out_g)
{

  __local float w_buffer[6][100];
  __local float im_buffer[100];
  __local float op_buffer[6];
  __local float bias_buffer[6];

  for (int i = 0; i < 6; i++) 
  {
    bias_buffer[i] = b_g[i];
  }

  //ELU
  for (int i = 0; i < 100; i++)
    {
    if (im_g[i] < 0) 
      im_buffer[i] = exp(im_g[i]) - 1;
    else
      im_buffer[i] = im_g[i];
    }

  for (int i = 0; i < 6; i++) 
  {
    for (int j = 0; j < 100; j++) 
    {
      w_buffer[i][j] = w_g[100 * i + j];
    }
  }

float cache[100];
for(int j=0;j<6;j++){
	float shift_reg[8];
	for(int i=0;i<8;i++)
		shift_reg[i]=0;
#pragma unroll
	for(int i=0;i<100;i++)
	cache[i]=w_buffer[j][i]*im_buffer[i];

	for(int i=0;i<100;i++)
	{
		shift_reg[7]=shift_reg[0]+cache[i];
		for(int i = 0; i < 7; ++i)
		{
			shift_reg[i] = shift_reg[i + 1];
		}
	}
	for(int i=0;i<7;i++){
		op_buffer[j]+=shift_reg[i];
	}
}

  for (int i = 0; i < 6 ; i++) 
  {
    op_buffer[i] = op_buffer[i]+ bias_buffer[i];
  }

 for (int i = 0; i < 6 ; i++) 
  {
   out_g[i] = op_buffer[i] ;
  }
  
}
/*******************************************stn_func*********************************************************/


// STN kernel

__constant float increment = 0.0625;
__constant float increment1 = -0.968750;

__kernel void stn_func(__global float * restrict  im_g,
		       __global float * restrict  theta_g,
                       __global float * restrict  output_g

) {

  // Local memory
  __local float im_flat[1024];
  __local float output[1024];
  float theta[2][3];
  __local float x[1024];
  __local float y[1024];

  __local float T[3][1024];
  __local float T_g[2][1024]; // matmul output
  float a[32];

  // Loading from Global memory to local memory

  for (int i = 0; i < 1024; i++) {
    im_flat[i] = im_g[i];
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      theta[i][j] = theta_g[3 * i + j];
    }
  }

  // meshgrid(T);
  // for(int i = 0 ;i < 32 ; i++)
  //  {
  //    a[i] = 0;
  //  }

  //  flatten_x(x_flat);
  //  linspace(a)

  a[0] = increment1;

  for (int i = 1; i < 32; i++) {
    a[i] = increment1 + i*increment;
  }

  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      T[0][i * 32 + j] = a[j];
      T[1][i * 32 + j] = a[i];
      T[2][i * 32 + j] = 1;
    }
  }

  // Mesh Grid Ended.

  // Matrix Multiplication begins
  // matmul(theta, T, T_g);
  // T: 3*1024 theta: 2*3
  // theta*t dim: 2*1024(T_g)

  float ans;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 1024; k++) {
      ans = 0;
      for (int j = 0; j < 3; j++) {
        ans += theta[i][j] * T[j][k];
      }
      T_g[i][k] = ans;
    }
  }

  for (int i = 0; i < 1024; i++) {

    x[i] = T_g[0][i];
    y[i] = T_g[1][i];
  }

  // inter(x, y, im, output);
  float max1 = 31; // 40 -1

  __local int x0[1024], y0[1024];
  __local int x1[1024], y1[1024];


  // float base[1024];
  __local int idx_a[1024];
  __local int idx_b[1024];
  __local int idx_c[1024];
  __local int idx_d[1024];

  float Ia, Ib, Ic, Id, wa, wb, wc, wd;

  for (int i = 0; i < 1024; i++) 
  {
    x[i] = (1 + x[i]) * 16;
    y[i] = (1 + y[i]) * 16;

    // if (x[i] >= 0) {
    //   x0[i] = (int)(x[i]);
    // } else {
    //   x0[i] = (int)(x[i]) - 1;
    // }


    // if (y[i] >= 0) {
    //   y0[i] = (int)(y[i]);
    // } else {
    //   y0[i] = (int)(y[i]) - 1;
    // }

    x0[i]=floor(x[i]);
    y0[i]=floor(y[i]);
    x1[i]=x0[i]+1;
    y1[i]=y0[i]+1;
  }

  for (int i = 0; i < 1024; i++)
   {
    if (x0[i] < 0)
      x0[i] = 0;

    if (x0[i] > max1)
      x0[i] = max1;
  }

   for (int i = 0; i < 1024; i++)
   {
    if (x1[i] < 0)
      x1[i] = 0;

    if (x1[i] > max1)
      x1[i] = max1;
  }

  for (int i = 0; i < 1024; i++)
   {
    if (y0[i] < 0)
      y0[i] = 0;

    if (y0[i] > max1)
      y0[i] = max1;
  }

  for (int i = 0; i < 1024; i++)
   {
    if (y1[i] < 0)
      y1[i] = 0;

    if (y1[i] > max1)
      y1[i] = max1;
  }

  for (int i = 0; i < 1024; i++) 
  {
    idx_a[i] = y0[i] * 32 + x0[i];
    idx_b[i] = y1[i] * 32 + x0[i];
    idx_c[i] = y0[i] * 32 + x1[i];
    idx_d[i] = y1[i] * 32 + x1[i];
  }

  // flatten(im,im_flat);

  for (int i = 0; i < 1024; i++)
   {
    Ia = im_flat[idx_a[i]];
    Ib = im_flat[idx_b[i]];
    Ic = im_flat[idx_c[i]];
    Id = im_flat[idx_d[i]];
    wa = (x1[i] - x[i]) * (y1[i] - y[i]);
    wb = (x1[i] - x[i]) * (y[i] - y0[i]);
    wc = (x[i] - x0[i]) * (y1[i] - y[i]);
    wd = (x[i] - x0[i]) * (y[i] - y0[i]);

    output[i] = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id);
  }

  for (int i = 0; i < 1024; i++) {
    output_g[i] = output[i];
  }
}



/********************************************conv1***********************************/
// Implements conv + relu + maxpooling


 __constant int  mp_conv1_base[4] = {0,1,28,29} ;

__kernel void conv1 (__global float * restrict im_g,
	             __global float * restrict  w_g ,
	             __global float * restrict  b_g ,
		     __global float * restrict  out_g 
		    )  

{
// Image Size = 32*32*3
// Convolved with 5*5*3 (100) filters generating 100 output feature maps
// No zero padding , stride = 1

__local float im_buffer[1024];

// 32 - 5 + 1 = 28
// 28*28 = 784

__local int  toep_buffer[784][25];
__local float  filter_patch[25][100] ; //100 filters
__local float  op_buffer[784][100];
__local int m2[196][4] ;     //For maxpooling with stride 2
__local float bias[100];

__local float output_buf[19600] ;

// Registers
int base[25];

//Initializing output buffer 

// for (int i = 0 ;i < 784 ; i++)
//      {
//      for (int j = 0 ; j < 100; j++)
//      {
//        op_buffer[i][j]  = 0; 
//      }
//      }


for (int i = 0; i < 5 ; i++) 
    {
    for (int j = 0; j < 5 ; j ++)
	 {
	  base[i*5 + j] = 32*i + j ;
	 }
	}

  for(int i = 0 ; i < 100 ; i++)
	{
	  bias[i] = b_g[i] ;
	}



//Toep Buffer contains the addresses
//Toeplitz Computation
  for (int i = 0 ; i < 28 ; i++)
   {
  	for (int j = 0 ; j < 28 ; j++)
   	{
	 for (int k = 0 ; k < 25 ; k ++)
	     {
		toep_buffer[28*i + j][k] = i*32 + base[k] + j ;
	     }
	}
   }

//m2 buffer for max pooling
 for (int i = 0 ; i < 14 ; i++)
    {

    for (int j = 0 ; j < 14 ; j++)
     {
 	for (int k = 0; k < 4 ; k++)
	    {
	      m2[14*i + j][k] = mp_conv1_base[k] + 2*j + 56*i ;
	    }
    }
  }

   
  
	
	// Loading data into im_buffer from global memory channel wise.

	 for (int i = 0 ; i < 1024 ; i++)
		{
		 im_buffer[i] = im_g[i ];
		}
	
	// Creating Filter patch matrix 
	
	for (int  k = 0 ; k < 100 ; k ++)
	     {
	     for (int i = 0; i < 25 ; i++)
	     {
	        filter_patch[i][k] = w_g[25*k + i ];
	     }
	     }	
	
	// computing output buffer of convolution

	 for (int i = 0 ; i < 784 ; i ++)
	  {
	   for (int j = 0 ; j < 25 ; j++)
	   {
	    for (int k = 0 ; k < 100 ; k++)
	     {
	       op_buffer[i][k] += im_buffer[toep_buffer[i][j]]*filter_patch[j][k];
	     }
	   }

	  }
		
		
	  // Leaky Relu , adding bias 
	 for (int i = 0 ; i < 784 ; i ++)
	 {
	  for(int j = 0; j < 100 ; j ++)
	  {
	   if (op_buffer[i][j] < 0)
	    {	
		op_buffer[i][j] = exp(op_buffer[i][j] + bias[j] )-1;
	     }
	    else
	    {
	      op_buffer[i][j] = op_buffer[i][j] + bias[j] ;
	    }
	  }
	 }

	//Maxpooling 
 	float a1 , b1, c1, d1 ;
	float max1,max2 ;
	float max3;

	for(int k = 0 ;k< 100 ; k++)
	  {
	   for (int i = 0 ; i < 196 ; i++)
	   {
	    a1 = op_buffer[m2[i][0]][k]; 
	    b1 = op_buffer[m2[i][1]][k]; 
	    c1 = op_buffer[m2[i][2]][k]; 
	    d1 = op_buffer[m2[i][3]][k]; 

	    if (a1 > b1)
	    {
	    max1 = a1;
	    }
	    else 
	    {
	    max1 = b1;
	    }
	    if (c1 > d1)
	    {
	    max2 = c1;
	    }
	    else
	    {
	    max2 = d1;
	    }
	    if (max1 > max2)
	    {
	    max3 = max1;
	    }
	    else 
	    {
	    max3 = max2 ;
	    }

	    //  out_g[196*k + i] = max3;
	      output_buf[196*k + i] = max3;
	    }
	    }
	for (int i = 0 ; i < 19600 ; i++)
	{
		out_g[i] = output_buf[i] ;	
	}
	    
	}
	

/*******************************************bn1**************************************************/


__kernel void bn1 (
			     __global float* restrict im_g,
			     __global float* restrict mean,
			      __global float* restrict variance,
			      __global float* restrict  gamma,
			      __global float* restrict beta ,
			      __global float * restrict out_g
		  )


{

  __local float  image_mem[196*100] ;
  __local float  out_mem[196*100]; 

     for(int i = 0; i < 196*100; i++)
	   {
	    image_mem[i] = im_g[i] ;
	   }

    for (int  j = 0 ; j < 100; j ++)
	    {
	    for(int i = 0 ; i < 196; i++)
             {
		out_mem[196*j + i] = gamma[j]*((image_mem[196*j + i] - mean[j] ) / sqrt(variance[j])) + beta[j] ;

	    }
	    }
		
	  for (int i = 0; i < 19600 ; i++)
	     {
		out_g[i] = out_mem[i] ;
	     }
	     
}



/*************************************************conv2******************************/

// Implements conv + relu + maxpooling

__constant int mp_conv2_base[4] = {0, 1, 12, 13};

__kernel void conv2(__global float * restrict  im_g,
                    __global float * restrict  w_g,
                    __global float * restrict  b_g, 
                    __global float * restrict  out_g)

{
  // Image Size = 32*32*3
  // Convolved with 5*5*3 (100) filters generating 100 output feature maps
  // No zero padding , stride = 1

  __local float im_buffer[196];

  // 32 - 5 + 1 = 28
  // 28*28 = 784

  __local int toep_buffer[144][9];
  __local float filter_patch[9][150]; // 100 filters
  __local float op_buffer[144][150];
  __local int m2[36][4]; // For maxpooling with stride 2
  __local float bias[150];

  // Registers
  int base[9];

  // Initializing output buffer

//   for (int i = 0; i < 144; i++) {
//     for (int j = 0; j < 150; j++) {
//       op_buffer[i][j] = 0;
//     }
//   }

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      base[i * 3 + j] = 14 * i + j;
    }
  }

  for (int i = 0; i < 150; i++) {
    bias[i] = b_g[i];
  }

  // Toep Buffer contains the addresses
  // Toeplitz Computation
  for (int i = 0; i < 12; i++) {
    for (int j = 0; j < 12; j++) {
      for (int k = 0; k < 9; k++) {
        toep_buffer[12 * i + j][k] = i * 14 + base[k] + j;
      }
    }
  }

  // m2 buffer for max pooling
  for (int i = 0; i < 6; i++) {

    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 4; k++) {
        m2[6 * i + j][k] = mp_conv2_base[k] + 2 * j + 24 * i;
      }
    }
  }

  // Channel Loop

  for (int c = 0; c < 100; c++) {

    // Loading data into im_buffer from global memory channel wise.

    for (int i = 0; i < 196; i++) {
      im_buffer[i] = im_g[196 * c + i];
    }

    // Creating Filter patch matrix

    for (int k = 0; k < 150; k++) {
      for (int i = 0; i < 9; i++) {
        filter_patch[i][k] = w_g[900 * k + 9 * c + i];
      }
    }

    // computing output buffer of convolution

    for (int i = 0; i < 144; i++) {
      for (int j = 0; j < 9; j++) {
        for (int k = 0; k < 150; k++) {
          op_buffer[i][k] += im_buffer[toep_buffer[i][j]] * filter_patch[j][k];
        }
      }
    }
  }

  // Leaky Relu , adding bias
  for (int i = 0; i < 144; i++) {
    for (int j = 0; j < 150; j++) {
      if (op_buffer[i][j] < 0) {
        op_buffer[i][j] = exp(op_buffer[i][j] + bias[j]) - 1;
      } else {
        op_buffer[i][j] = op_buffer[i][j] + bias[j];
      }
    }
  }

  // Maxpooling
  float a1, b1, c1, d1;
  float max1, max2;
  float max3;

  for (int k = 0; k < 150; k++) {
    for (int i = 0; i < 36; i++) {
      a1 = op_buffer[m2[i][0]][k];
      b1 = op_buffer[m2[i][1]][k];
      c1 = op_buffer[m2[i][2]][k];
      d1 = op_buffer[m2[i][3]][k];

      if (a1 > b1) {
        max1 = a1;
      } else {
        max1 = b1;
      }
      if (c1 > d1) {
        max2 = c1;
      } else {
        max2 = d1;
      }
      if (max1 > max2) {
        max3 = max1;
      } else {
        max3 = max2;
      }

      out_g[36 * k + i] = max3;
    }
  }
}




		
/********************************************************bn2*********************************************************/



//pipe float p0 __attribute__((xcl_reqd_pipe_depth(32768)));

__kernel void bn2(
			     __global float* restrict   im_g,
			     __global float* restrict  mean,
			      __global float* restrict  variance,
			      __global float* restrict  gamma,
			      __global float* restrict  beta ,
			      __global float* restrict  out
			    )


{

  __local float  image_mem[5400] ;
  __local float  out_mem[5400]; 

     for(int i = 0; i < 5400 ; i++)
	   {
	    image_mem[i] = im_g[i] ;
	   
	   }

    for (int  j = 0 ; j < 150; j ++)
	    {
	    for(int i = 0 ; i < 36; i++)
             {
		out_mem[36*j + i] = gamma[j]*((image_mem[36*j + i] - mean[j] ) / sqrt(variance[j])) + beta[j] ;

	    }
	  }

	for(int i = 0 ; i < 5400 ; i++)
	 {
	 out[i] = out_mem[i] ;
	 }

}


/*************************************conv3*****************************************************/


 __constant int  mp_conv3_base[4] = {0,1,6,7} ;

__kernel void conv3 (__global float * restrict im_g,
	             __global float *  restrict     w_g ,
	             __global float *  restrict     b_g ,
		     __global float *      restrict     out_g 
		    )  

{
__local float im_buffer[36];
__local float  filter_patch[250] ; //100 filters
__local float  op_buffer[36][250];
__local int m2[9][4];     //For maxpooling with stride 2
__local float bias[250];


//Initializing output buffer 

// for (int i = 0 ;i < 36 ; i++)
//      {
//      for (int j = 0 ; j < 250; j++)
//      {
//        op_buffer[i][j]  = 0; 
//      }
//      }


  for(int i = 0 ; i < 250 ; i++)
	{
	  bias[i] = b_g[i] ;
	}


//m2 buffer for max pooling
 for (int i = 0 ; i < 3 ; i++)
    {
    for (int j = 0 ; j < 3 ; j++)
     {
 	for (int k = 0; k < 4 ; k++)
	    {
	      m2[3*i + j][k] = mp_conv3_base[k] + 2*j + 12*i ;
	    }
    }
  }

  //Channel Loop
   
  for (int c = 0 ; c < 150 ;c++)
	 {
	
	// Loading data into im_buffer from global memory channel wise.

	 for (int i = 0 ; i < 36 ; i++)
		{
		 im_buffer[i] = im_g[36*c + i ];
		}
	
	// Creating Filter patch matrix 
	
	for (int  k = 0 ; k < 250 ; k ++)
	     {
	        filter_patch[k] = w_g[150*k + c];
	     }
	
	 for (int i = 0 ; i < 36 ; i ++)
		{	
	    for (int k = 0 ; k < 250 ; k++) 
	       op_buffer[i][k] += im_buffer[i]*filter_patch[k];
	    }
	 }

		
	  // Leaky Relu , adding bias 
	 for (int i = 0 ; i < 36 ; i ++)
	 {
	  for(int j = 0; j < 250 ; j ++)
	  {
	   if (op_buffer[i][j] < 0)
	     {
		op_buffer[i][j] = exp(op_buffer[i][j] + bias[j] )-1;
	     }
	    else
	    {
	      op_buffer[i][j] = op_buffer[i][j] + bias[j] ;
	    }
	  }
	 }

	//Maxpooling 
 	float a1 , b1, c1, d1 ;
	float max1,max2 ;
	float max3;

	for(int k = 0 ;k< 250 ; k++)
	  {
	   for (int i = 0 ; i < 9 ; i++)
	   {
	    a1 = op_buffer[m2[i][0]][k]; 
	    b1 = op_buffer[m2[i][1]][k]; 
	    c1 = op_buffer[m2[i][2]][k]; 
	    d1 = op_buffer[m2[i][3]][k]; 

	    if (a1 > b1)
	    {
	    max1 = a1;
	    }
	    else 
	    {
	    max1 = b1;
	    }
	    if (c1 > d1)
	    {
	    max2 = c1;
	    }
	    else
	    {
	    max2 = d1;
	    }
	    if (max1 > max2)
	    {
	    max3 = max1;
	    }
	    else 
	    {
	    max3 = max2 ;
	    }
	      out_g[9*k + i] = max3;
	    }
	    	
	  }
	}
		
		
/**************************************************bn3****************************************/




__kernel void bn3 (
			     __global float* restrict im_g,
			     __global float* restrict mean,
			      __global float* restrict variance,
			      __global float* restrict  gamma,
			      __global float* restrict beta ,
			      __global float * restrict out
			    )


{

  __local float  image_mem[2250] ;
  __local float  out_mem[2250]; 

     for(int i = 0; i < 2250; i++)
	   {
	    image_mem[i] = im_g[i] ;
	   
	   }

    for (int  j = 0 ; j < 250; j ++)
	    {
	    for(int i = 0 ; i < 9; i++)
             {
		out_mem[9*j + i] = gamma[j]*((image_mem[9*j + i] - mean[j] ) / sqrt(variance[j])) + beta[j] ;

	    }
	  }
	  for(int i = 0; i < 2250 ; i++)
	   {
		
		out[i] = out_mem[i] ;	
	
	   }
	  
	  }


/*******************************FC3.cl***********************/

__constant int W_FC3_height = 350 ;
__constant int W_FC3_width  = 2250 ;
__constant int X_FC3_width = 1 ;




//__attribute__((reqd_work_group_size(1,16,1)))

__kernel void FC3 (    __global float *restrict X,
		       __global float *restrict W,
		       __global float *restrict bias,
		       __global float *restrict Y
		  )



{

const int global_x = get_global_id(0);    //globadXDx of a work item
const int global_y = get_global_id(1);    //globadXDy of a work item


float  acc = 0.0f;

for (int k=0 ; k<W_FC3_width ; k++) 
{
acc += W[global_y*W_FC3_width + k]*X[k*X_FC3_width + global_x];
}

//Store the result.
Y[global_y*X_FC3_width + global_x] =acc + bias[global_y*X_FC3_width + global_x] ;
}

/******************************************bn4*******************************/

__kernel void bn4 (
			     __global float* restrict im_g,
			     __global float* restrict mean,
			      __global float* restrict variance,
			      __global float* restrict  gamma,
			      __global float* restrict beta ,
			      __global float * restrict out
			    )


{

  __local float  image_mem[350] ;
  __local float  out_mem[350]; 

     for(int i = 0; i < 350; i++)
	   {
	 //  read_pipe_block(p0,&image_mem[i]);
	    image_mem[i] = im_g[i] ;
	   
	   }

    for (int  j = 0 ; j < 350; j ++)
	    {
		out_mem[j] = gamma[j]*((image_mem[j] - mean[j] ) / sqrt(variance[j])) + beta[j] ;

	    }
	  
	for(int i = 0; i < 350 ; i++)
	 {
		out[i] = out_mem[i] ;
     }



}

/**********************************FC4.cl***************************************/
__kernel void FC4(
                   __global float * restrict  im_g,
	               __global float * restrict w_g,
                   __global float * restrict bias_g,
		            __global float * restrict op_g
                 )
  {
    
    __local float im[350];
    __local float w[43][350];
    __local float op[43];
    __local float bias[43];

      float max_val = 0;
      float index ;

    for(int i=0;i<350;i++){
        im[i]=im_g[i];
    }

    for(int i=0;i<43;i++){
        bias[i]=bias_g[i];
        for(int j=0;j<350;j++){
            w[i][j]=w_g[350*i+j];
        }
    }

float cache[43];   
for(int j=0;j<350;j++){
    for(int i=0;i<43;i++){    
   				op[i]+=w[i][j]*im[j];
			}
        }
    

    for(int i=0;i<43;i++){
        op[i]=op[i]+bias[i];
   	op[i] = exp(op[i]) ; 
       	if ((max_val) < op[i])
	{	max_val = op[i];
		index = i;

	}
	}
	
     op_g[0] = index ; 

}





