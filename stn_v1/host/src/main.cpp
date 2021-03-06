//STN host
//Authors : Prithvi , Sivani

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "weights/im_32.h"
#include "weights/im1.h"
#include "weights/im2.h"
#include "weights/im3.h"
#include "weights/im4.h"
#include "weights/im5.h"
#include "weights/im6.h"
#include "weights/im7.h"
#include "weights/im8.h"
#include "weights/im9.h"





#include "weights/stn.loc_net.0.weight.h"
#include "weights/stn.loc_net.0.bias.h"

#include "weights/stn.loc_net.3.weight.h"
#include "weights/stn.loc_net.3.bias.h"

#include "weights/stn.fc_loc.0.weight.h"
#include "weights/stn.fc_loc.0.bias.h"

#include "weights/stn.fc_loc.2.weight.h"
#include "weights/stn.fc_loc.2.bias.h"


#include "weights/conv1.weight.h"
#include "weights/conv1.bias.h"

#include "weights/conv1_bn.bias.h"
#include "weights/conv1_bn.weight.h"
#include "weights/conv1_bn.running_mean.h"
#include "weights/conv1_bn.running_var.h"


#include "weights/conv2.weight.h"
#include "weights/conv2.bias.h"

#include "weights/conv2_bn.bias.h"
#include "weights/conv2_bn.weight.h"
#include "weights/conv2_bn.running_mean.h"
#include "weights/conv2_bn.running_var.h"


#include "weights/conv3.weight.h"
#include "weights/conv3.bias.h"

#include "weights/conv3_bn.bias.h"
#include "weights/conv3_bn.weight.h"
#include "weights/conv3_bn.running_mean.h"
#include "weights/conv3_bn.running_var.h"

#include "weights/fc1.weight.h"
#include "weights/fc1.bias.h"

#include "weights/fc1_bn.bias.h"
#include "weights/fc1_bn.weight.h"
#include "weights/fc1_bn.running_mean.h"
#include "weights/fc1_bn.running_var.h"


#include "weights/fc2.weight.h"
#include "weights/fc2.bias.h"







using namespace aocl_utils;

static cl_platform_id platform = NULL;
static cl_device_id device; // num_devices elements
static cl_context context = NULL;
static cl_command_queue queue;
static cl_program program = NULL;
static cl_kernel kernel[14]  ; 



// Problem data.
unsigned  N1 = 1024 ;
unsigned  N2 = 7*7*50 ;
unsigned  N3 = 50 ;
unsigned  N4 = 13*13*50 ;
unsigned  N5 = 5*5*50*100 ;
unsigned  N6 = 100;
unsigned  N7 = 1600;
unsigned  N8 = 1600*100 ;
unsigned  N9 = 600;
unsigned  N10 = 6;
unsigned  N11 = 2500;
unsigned  N12 = 14*14*100 ;
unsigned  N13 = 150*100*3*3 ;
unsigned  N14 = 150;
unsigned  N15 = 6*6*150 ;
unsigned  N16 = 250*150 ;
unsigned  N17 = 250;
unsigned  N18 = 3*3*250 ;
unsigned  N19 = 250*3*3*350 ;
unsigned  N20 = 350;
unsigned  N21 = 350*43 ;
unsigned  N22 = 43;
unsigned  N23 = 1;
unsigned  N24 = 1 ;



/*Aligned Memory*/

//stn1_conv1
float *source_image , *stn_conv1_wt , *stn_conv1_bias  ;
cl_mem image_buf,stn_conv1_wt_buf, stn_conv1_bias_buf , inout_buf ; 

//stn_conv2 
float *stn_conv2_wt, *stn_conv2_bias ;
cl_mem stn_conv2_wt_buf, stn_conv2_bias_buf , inout1_buf ; 

//FC1
float *fc1_wt ,*fc1_bias ;
cl_mem fc1_wt_buf , fc1_bias_buf, inout2_buf ; 


//FC2
float *fc2_wt ,*fc2_bias ;
cl_mem fc2_wt_buf , fc2_bias_buf, theta_buf ; 

//stn_func

cl_mem inout3_buf ;

//conv1
float *conv1_wt, *conv1_bias ;
cl_mem conv1_wt_buf,conv1_bias_buf, inout4_buf;

//bn1
float *bn1_mean ,*bn1_variance ,*bn1_gamma ,*bn1_beta ;
cl_mem bn1_mean_buf , bn1_variance_buf, bn1_gamma_buf, bn1_beta_buf, inout5_buf ;

//conv2
float *conv2_wt, *conv2_bias ;
cl_mem conv2_wt_buf,conv2_bias_buf, inout6_buf;

//bn2
float *bn2_mean ,*bn2_variance ,*bn2_gamma ,*bn2_beta ;
cl_mem  bn2_mean_buf ,bn2_variance_buf, bn2_gamma_buf, bn2_beta_buf, inout7_buf ;

//conv3
float *conv3_wt, *conv3_bias ;
cl_mem conv3_wt_buf,conv3_bias_buf, inout8_buf;

//bn3
float *bn3_mean ,*bn3_variance ,*bn3_gamma ,*bn3_beta ;
cl_mem bn3_mean_buf ,bn3_variance_buf, bn3_gamma_buf, bn3_beta_buf, inout9_buf ;

//FC3
float *fc3_wt ,*fc3_bias ;
cl_mem fc3_wt_buf , fc3_bias_buf, inout10_buf ;

//bn4
float *bn4_mean ,*bn4_variance ,*bn4_gamma ,*bn4_beta ;
cl_mem bn4_mean_buf ,bn4_variance_buf, bn4_gamma_buf, bn4_beta_buf, inout11_buf;

//FC4
float *fc4_wt ,*fc4_bias,*output ;
cl_mem fc4_wt_buf, fc4_bias_buf, output_buf ; 

float *debug_output ;
float load_image[1024];

//Emulator
bool use_emulator = 0  ;


//Prototypes
float rand_float();
bool init_opencl();
void init_problem();
void run();
void cleanup();

// Entry point.
int main(int argc, char *argv[]) {

    // Initialize OpenCL.
    if(!init_opencl()) {
        return -1;
    }

    char *a = argv[1] ;
    int element = atoi(a);

    if (argc != 2)
    {
     printf("\nUsage bin/host arg");
     exit(0);
    }
     
    switch(element)
   {
    case 0:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image_h[i];
    }
    printf("\nLoading image0\n");
    break ;

    case 1:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image1_h[i];
    }
    printf("\nLoading image1\n");
    break ;

    case 2:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image2_h[i];
    }
    printf("\nLoading image2\n");
    break;

    case 3:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image3_h[i];
    }
    printf("\nLoading image3\n");
    break;

    case 4:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image4_h[i];
    }
    printf("\nLoading image4\n");
    break;


    case 5:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image5_h[i];
    }
    printf("\nLoading image5\n");
    break;

    case 6:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image6_h[i];
    }
    printf("\nLoading image6\n");
    break;
   
    case 7:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image7_h[i];
    }
    printf("\nLoading image7\n");
    break;

    case 8:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image8_h[i];
    }
    printf("\nLoading image8\n");
    break;


    case 9:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image9_h[i];
    }
    printf("\nLoading image9\n");
    break;


    default:
    for (unsigned i = 0 ;i < 1024 ; i++)
    {
    load_image[i] = source_image_h[i];
    }
    printf("\nLoading image0\n");

  }

    // Initialize the problem data.
    init_problem();

    // Run the kernel.Control
    run();

    // Free the resources allocated
    cleanup();

    return 0;
}

/////// HELPER FUNCTIONS ///////

// Randomly generate a floating-point number between 0 to 1.
float rand_float() {
    return float(rand()) / float(RAND_MAX)  ;
}

// Initializes the OpenCL objects.
bool init_opencl() {
    cl_int status;

    printf("\nInitializing OpenCL\n");

    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.


    if (use_emulator) {
        platform = findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    }

    else{
        platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        return false;
    } 

   /*       platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
          if(platform == NULL) {
          printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
          return false;
          } */ 

    // Query the available OpenCL device.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    //We ll use only one device
    device = devices[0];

    printf("Platform: %s\n", getPlatformName(platform).c_str());

    // Create the context.
    context = clCreateContext(NULL, 1,&device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    //Create the command queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    //Create the program 
    std::string binary_file = getBoardBinaryFile("traffic_stn", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(),&device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the kernel
    const char *kernel_name0 = "stn_conv1";
    kernel[0] = clCreateKernel(program, kernel_name0, &status);
    checkError(status, "Failed to create kernel for stn_conv1");

    const char *kernel_name1 = "stn_conv2";
    kernel[1] = clCreateKernel(program, kernel_name1, &status);
    checkError(status, "Failed to create kernel for stn_conv2");

    const char *kernel_name2 = "FC1";
    kernel[2] = clCreateKernel(program, kernel_name2, &status);
    checkError(status, "Failed to create kernel for  FC1");

    const char *kernel_name3 = "FC2";
    kernel[3] = clCreateKernel(program, kernel_name3, &status);
    checkError(status, "Failed to create kernel for FC2");

    const char *kernel_name4 = "stn_func";
    kernel[4] = clCreateKernel(program, kernel_name4, &status);
    checkError(status, "Failed to create kernel for stn_func");

    const char *kernel_name5 = "conv1";
    kernel[5] = clCreateKernel(program, kernel_name5, &status);
    checkError(status, "Failed to create kernel for conv1");


    const char *kernel_name6 = "bn1";
    kernel[6] = clCreateKernel(program, kernel_name6, &status);
    checkError(status, "Failed to create kernel for bn1");

    const char *kernel_name7 = "conv2";
    kernel[7] = clCreateKernel(program, kernel_name7, &status);
    checkError(status, "Failed to create kernel for conv2");


    const char *kernel_name8 = "bn2";
    kernel[8] = clCreateKernel(program, kernel_name8, &status);
    checkError(status, "Failed to create kernel for bn2");



    const char *kernel_name9 = "conv3";
    kernel[9] = clCreateKernel(program, kernel_name9, &status);
    checkError(status, "Failed to create kernel for conv3");


    const char *kernel_name10 = "bn3";
    kernel[10] = clCreateKernel(program, kernel_name10, &status);
    checkError(status, "Failed to create kernel for bn3");

    const char *kernel_name11 = "FC3";
    kernel[11] = clCreateKernel(program, kernel_name11, &status);
    checkError(status, "Failed to create kernel for FC3");


    const char *kernel_name12 = "bn4";
    kernel[12] = clCreateKernel(program, kernel_name12, &status);
    checkError(status, "Failed to create kernel for bn4");

    const char *kernel_name13 = "FC4";
    kernel[13] = clCreateKernel(program, kernel_name13, &status);
    checkError(status, "Failed to create kernel for FC4");



    //Creating buffers
    //stn_conv1
    image_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N1 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for image");


    stn_conv1_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N2 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for stn_conv1_wt");


    stn_conv1_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N3 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for stn_conv1_bias");

    inout_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N4 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out");

    //stn_conv2

    stn_conv2_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N5 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for stn_conv2_wt");


    stn_conv2_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for stn_conv2_bias");

    inout1_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N7 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out1");

    //FC1

    fc1_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N8 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc1_wt");

    fc1_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc1_bias");

    inout2_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out2");

    //FC2

    fc2_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N9 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc2_wt");

    fc2_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N10 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc2_bias");

    theta_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N10 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for theta_buf");

    //stn_func

    inout3_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N1 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out3");

    //conv1

    conv1_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N11 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv1_wt");

    conv1_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv1_bias");

    inout4_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N12 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out4");

    //bn1 

    bn1_mean_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn1_mean");

    bn1_variance_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn1_variance");

    bn1_gamma_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn1_gamma");


    bn1_beta_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N6 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn1_beta");


    inout5_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N12 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out5");


    //conv2

    conv2_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N13 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv2_wt");

    conv2_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N14 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv2_bias");

    inout6_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N15 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out6");


    //bn2 

    bn2_mean_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N14 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn2_mean");

    bn2_variance_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N14 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn2_variance");

    bn2_gamma_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N14 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn2_gamma");

    bn2_beta_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N14 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn2_beta");


    inout7_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N15 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out7");


    //conv3

    conv3_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N16 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv3_wt");

    conv3_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N17 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for conv3_bias");

    inout8_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N18 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out8");


    //bn3 

    bn3_mean_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N17 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn3_mean");

    bn3_variance_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N17 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn3_variance");

    bn3_gamma_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N17 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn3_gamma");

    bn3_beta_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N17 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn3_beta");


    inout9_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N18 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out9");

    //FC3

    fc3_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N19 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc3_wt");

    fc3_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc3_bias");

    inout10_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out10");


    //bn4 

    bn4_mean_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn4_mean");

    bn4_variance_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn4_variance");

    bn4_gamma_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn4_gamma");

    bn4_beta_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for bn4_beta");


    inout11_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N20 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for in_out11");

    //FC4

    fc4_wt_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N21 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc4_wt");

    fc4_bias_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            N22 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for fc4_bias");

    output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            N23 * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");

    return true;
}

// Initialize the data for the problem.
void init_problem() {

    //stn_conv1
    posix_memalign ((void **)(&source_image),64,sizeof(float)*N1);
    posix_memalign ((void **)(&stn_conv1_wt),64,sizeof(float)*N2);
    posix_memalign ((void **)(&stn_conv1_bias),64,sizeof(float)*N3);


    posix_memalign ((void **)(&debug_output),64,sizeof(float)*N24);
    //stn_conv2
    posix_memalign ((void **)(&stn_conv2_wt),64,sizeof(float)*N5);
    posix_memalign ((void **)(&stn_conv2_bias),64,sizeof(float)*N6);


    //FC1
    posix_memalign ((void **)(&fc1_wt),64,sizeof(float)*N8);
    posix_memalign ((void **)(&fc1_bias),64,sizeof(float)*N6);

    //FC2
    posix_memalign ((void **)(&fc2_wt),64,sizeof(float)*N9);
    posix_memalign ((void **)(&fc2_bias),64,sizeof(float)*N10);

    //stn_func

    //conv1
    posix_memalign ((void **)(&conv1_wt),64,sizeof(float)*N11);
    posix_memalign ((void **)(&conv1_bias),64,sizeof(float)*N6);

    //bn1
    posix_memalign ((void **)(&bn1_mean),64,sizeof(float)*N6);
    posix_memalign ((void **)(&bn1_variance),64,sizeof(float)*N6);
    posix_memalign ((void **)(&bn1_gamma),64,sizeof(float)*N6);
    posix_memalign ((void **)(&bn1_beta),64,sizeof(float)*N6);

    //conv2
    posix_memalign ((void **)(&conv2_wt),64,sizeof(float)*N13);
    posix_memalign ((void **)(&conv2_bias),64,sizeof(float)*N14);

    //bn2
    posix_memalign ((void **)(&bn2_mean),64,sizeof(float)*N14);
    posix_memalign ((void **)(&bn2_variance),64,sizeof(float)*N14);
    posix_memalign ((void **)(&bn2_gamma),64,sizeof(float)*N14);
    posix_memalign ((void **)(&bn2_beta),64,sizeof(float)*N14);

    //conv3
    posix_memalign ((void **)(&conv3_wt),64,sizeof(float)*N16);
    posix_memalign ((void **)(&conv3_bias),64,sizeof(float)*N17);

    //bn3
    posix_memalign ((void **)(&bn3_mean),64,sizeof(float)*N17);
    posix_memalign ((void **)(&bn3_variance),64,sizeof(float)*N17);
    posix_memalign ((void **)(&bn3_gamma),64,sizeof(float)*N17);
    posix_memalign ((void **)(&bn3_beta),64,sizeof(float)*N17);

    //FC3
    posix_memalign ((void **)(&fc3_wt),64,sizeof(float)*N19);
    posix_memalign ((void **)(&fc3_bias),64,sizeof(float)*N20);

    //bn4
    posix_memalign ((void **)(&bn4_mean),64,sizeof(float)*N20);
    posix_memalign ((void **)(&bn4_variance),64,sizeof(float)*N20);
    posix_memalign ((void **)(&bn4_gamma),64,sizeof(float)*N20);
    posix_memalign ((void **)(&bn4_beta),64,sizeof(float)*N20);

    //FC4
    posix_memalign ((void **)(&fc4_wt),64,sizeof(float)*N21);
    posix_memalign ((void **)(&fc4_bias),64,sizeof(float)*N22);
    posix_memalign ((void **)(&output),64,sizeof(float)*N23);


    //initialize random floating values for image	

    //stn_conv1	
    for(unsigned j = 0; j < N1; j++) {
        source_image[j] = load_image[j] ;
    }

    for(unsigned j = 0; j < N2; j++) {
        stn_conv1_wt[j] = stn_conv1_wt_h[j] ;
    }

    for(unsigned j = 0; j < N3; j++) {
        stn_conv1_bias[j] = stn_conv1_bias_h[j] ;
    }
    //stn_conv2

    for(unsigned j = 0; j < N5; j++) {
        stn_conv2_wt[j] = stn_conv2_wt_h[j] ;
    }

    for(unsigned j = 0; j < N6; j++) {
        stn_conv2_bias[j] = stn_conv2_bias_h[j] ;
    }

    //FC1

    for(unsigned j = 0; j < N8 ; j++) {
        fc1_wt[j] = fc1_wt_h[j] ;
    }

    for(unsigned j = 0; j < N6 ; j++) {
        fc1_bias[j] = fc1_bias_h[j] ;
    }

    //FC2
    for(unsigned j = 0; j < N9 ; j++) {
        fc2_wt[j] = fc2_wt_h[j] ;
    }

    for(unsigned j = 0; j < N10 ; j++) {
        fc2_bias[j] = fc2_bias_h[j] ;
    }
    //stn_func

    //conv1
    for(unsigned j = 0; j < N11 ; j++) {
        conv1_wt[j] = conv1_wt_h[j] ;
    }

    for(unsigned j = 0; j < N6 ; j++) {
        conv1_bias[j] = conv1_bias_h[j] ;
    }


    //bn1
    for(unsigned j = 0; j < N6 ; j++) {
        bn1_mean[j] =     bn1_mean_h[j];
        bn1_variance[j] = bn1_variance_h[j];
        bn1_gamma[j] =    bn1_gamma_h[j];
        bn1_beta[j] =     bn1_beta_h[j];
    }

    //conv2
    for(unsigned j = 0; j < N13 ; j++) {
        conv2_wt[j] = conv2_wt_h[j] ;
    }

    for(unsigned j = 0; j < N14 ; j++) {
        conv2_bias[j] = conv2_bias_h[j] ;
    }

    //bn2
    for(unsigned j = 0; j < N14 ; j++) {
        bn2_mean[j] =     bn2_mean_h[j];
        bn2_variance[j] = bn2_variance_h[j];
        bn2_gamma[j] =    bn2_gamma_h[j];
        bn2_beta[j] =     bn2_beta_h[j];
    }

    //conv3
    for(unsigned j = 0; j < N16 ; j++) {
        conv3_wt[j] = conv3_wt_h[j] ;
    }

    for(unsigned j = 0; j < N17 ; j++) {
        conv3_bias[j] = conv3_bias_h[j] ;
    }

    //bn3
    for(unsigned j = 0; j < N17 ; j++) {
        bn3_mean[j] =     bn3_mean_h[j];
        bn3_variance[j] = bn3_variance_h[j];
        bn3_gamma[j] =    bn3_gamma_h[j];
        bn3_beta[j] =     bn3_beta_h[j];
    }

    //FC3
    for(unsigned j = 0; j < N19 ; j++) {
        fc3_wt[j] = fc3_wt_h[j] ;
    }

    for(unsigned j = 0; j < N20 ; j++) {
        fc3_bias[j] = fc3_bias_h[j] ;
    }

    //bn4
    for(unsigned j = 0; j < N20 ; j++) {
        bn4_mean[j] =     bn4_mean_h[j];
        bn4_variance[j] = bn4_variance_h[j];
        bn4_gamma[j] =    bn4_gamma_h[j];
        bn4_beta[j] =     bn4_beta_h[j];
    }

    //FC4
    for(unsigned j = 0; j < N21 ; j++) {
        fc4_wt[j] = fc4_wt_h[j] ;
    }

    for(unsigned j = 0; j < N22 ; j++) {
        fc4_bias[j] = fc4_bias_h[j] ;
    } 

}

void run() {
    cl_int status;

    const double start_time = getCurrentTimestamp();


    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.


    //stn_conv1  
    status = clEnqueueWriteBuffer(queue, image_buf, CL_FALSE,
            0, N1 * sizeof(float), source_image, 0, NULL, NULL);
    checkError(status, "Failed to transfer image_buf");

    status = clEnqueueWriteBuffer(queue, stn_conv1_wt_buf, CL_FALSE,
            0, N2 * sizeof(float), stn_conv1_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer stn_conv1_wt_buf");

    status = clEnqueueWriteBuffer(queue, stn_conv1_bias_buf, CL_FALSE,
            0, N3 * sizeof(float), stn_conv1_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer stn_conv1_bias_buf");


    //stn_conv2

    status = clEnqueueWriteBuffer(queue, stn_conv2_wt_buf, CL_FALSE,
            0, N5 * sizeof(float), stn_conv2_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer stn_conv2_wt_buf");

    status = clEnqueueWriteBuffer(queue, stn_conv2_bias_buf, CL_FALSE,
            0, N6 * sizeof(float), stn_conv2_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer stn_conv2_bias_buf");



    //FC1

    status = clEnqueueWriteBuffer(queue, fc1_wt_buf, CL_FALSE,
            0, N8 * sizeof(float), fc1_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc1_wt_buf");

    status = clEnqueueWriteBuffer(queue, fc1_bias_buf, CL_FALSE,
            0, N6 * sizeof(float), fc1_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc1_bias_buf");


    //FC2	

    status = clEnqueueWriteBuffer(queue, fc2_wt_buf, CL_FALSE,
            0, N9 * sizeof(float), fc2_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc2_wt_buf");

    status = clEnqueueWriteBuffer(queue, fc2_bias_buf, CL_FALSE,
            0, N10 * sizeof(float), fc2_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc2_bias_buf"); 

    //stn_func



    //conv1 

    status = clEnqueueWriteBuffer(queue, conv1_wt_buf, CL_FALSE,
            0, N11 * sizeof(float), conv1_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv1_wt_buf");

    status = clEnqueueWriteBuffer(queue, conv1_bias_buf, CL_FALSE,
            0, N6 * sizeof(float), conv1_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv1_bias_buf");

    //bn1
    status = clEnqueueWriteBuffer(queue, bn1_mean_buf, CL_FALSE,
            0, N6 * sizeof(float), bn1_mean, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn1_mean_buf");

    status = clEnqueueWriteBuffer(queue, bn1_variance_buf, CL_FALSE,
            0, N6 * sizeof(float), bn1_variance, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn1_variance_buf");

    status = clEnqueueWriteBuffer(queue, bn1_gamma_buf, CL_FALSE,
            0, N6 * sizeof(float), bn1_gamma, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn1_gamma_buf");

    status = clEnqueueWriteBuffer(queue, bn1_beta_buf, CL_FALSE,
            0, N6 * sizeof(float), bn1_beta, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn1_beta_buf");

    //conv2

    status = clEnqueueWriteBuffer(queue, conv2_wt_buf, CL_FALSE,
            0, N13 * sizeof(float), conv2_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv2_wt_buf");

    status = clEnqueueWriteBuffer(queue, conv2_bias_buf, CL_FALSE,
            0, N14 * sizeof(float), conv2_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv2_bias_buf");

    //bn2

    status = clEnqueueWriteBuffer(queue, bn2_mean_buf, CL_FALSE,
            0, N14 * sizeof(float), bn2_mean, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn2_mean_buf");

    status = clEnqueueWriteBuffer(queue, bn2_variance_buf, CL_FALSE,
            0, N14 * sizeof(float), bn2_variance, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn2_variance_buf");

    status = clEnqueueWriteBuffer(queue, bn2_gamma_buf, CL_FALSE,
            0, N14 * sizeof(float), bn2_gamma, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn2_gamma_buf");

    status = clEnqueueWriteBuffer(queue, bn2_beta_buf, CL_FALSE,
            0, N14 * sizeof(float), bn2_beta, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn2_beta_buf"); 

    //conv3

    status = clEnqueueWriteBuffer(queue, conv3_wt_buf, CL_FALSE,
            0, N16 * sizeof(float), conv3_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv3_wt_buf");

    status = clEnqueueWriteBuffer(queue, conv3_bias_buf, CL_FALSE,
            0, N17 * sizeof(float), conv3_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer conv3_bias_buf");


    //bn3

    status = clEnqueueWriteBuffer(queue, bn3_mean_buf, CL_FALSE,
            0, N17 * sizeof(float), bn3_mean, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn3_mean_buf");

    status = clEnqueueWriteBuffer(queue, bn3_variance_buf, CL_FALSE,
            0, N17 * sizeof(float), bn3_variance, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn3_variance_buf");

    status = clEnqueueWriteBuffer(queue, bn3_gamma_buf, CL_FALSE,
            0, N17 * sizeof(float), bn3_gamma, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn3_gamma_buf");

    status = clEnqueueWriteBuffer(queue, bn3_beta_buf, CL_FALSE,
            0, N17 * sizeof(float), bn3_beta, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn3_beta_buf"); 




    //FC3	

    /*  status = clEnqueueWriteBuffer(queue, fc3_wt_buf, CL_FALSE,
        0, N19 * sizeof(float), fc3_wt, 0, NULL, &write_event[0]);
        checkError(status, "Failed to transfer fc3_wt_buf");

        status = clEnqueueWriteBuffer(queue, fc3_bias_buf, CL_FALSE,
        0, N20 * sizeof(float), fc3_bias, 0, NULL, &write_event[1]);
        checkError(status, "Failed to transfer fc3_bias_buf"); */


    status = clEnqueueWriteBuffer(queue, fc3_wt_buf, CL_FALSE,
            0, N19 * sizeof(float), fc3_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc3_wt_buf");

    status = clEnqueueWriteBuffer(queue, fc3_bias_buf, CL_FALSE,
            0, N20 * sizeof(float), fc3_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc3_bias_buf"); 





    //bn4

    status = clEnqueueWriteBuffer(queue, bn4_mean_buf, CL_FALSE,
            0, N20 * sizeof(float), bn4_mean, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn4_mean_buf");

    status = clEnqueueWriteBuffer(queue, bn4_variance_buf, CL_FALSE,
            0, N20 * sizeof(float), bn4_variance, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn4_variance_buf");

    status = clEnqueueWriteBuffer(queue, bn4_gamma_buf, CL_FALSE,
            0, N20 * sizeof(float), bn4_gamma, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn4_gamma_buf");

    status = clEnqueueWriteBuffer(queue, bn4_beta_buf, CL_FALSE,
            0, N20 * sizeof(float), bn4_beta, 0, NULL, NULL);
    checkError(status, "Failed to transfer bn4_beta_buf"); 


    //FC4	

    status = clEnqueueWriteBuffer(queue, fc4_wt_buf, CL_FALSE,
            0, N21 * sizeof(float), fc4_wt, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc4_wt_buf");

    status = clEnqueueWriteBuffer(queue, fc4_bias_buf, CL_FALSE,
            0, N22 * sizeof(float), fc4_bias, 0, NULL, NULL);
    checkError(status, "Failed to transfer fc4_bias_buf"); 


    //Write events to all kernels.


   // clFinish(queue); 			//added
    cl_event kernel_event[14] ;

    // Set kernel arguments.
    //	unsigned argi = 0;

    //stn_conv1
    status = clSetKernelArg(kernel[0], 0, sizeof(cl_mem), &image_buf);
    checkError(status, "Failed to set argument 0 for stn_conv1");

    status = clSetKernelArg(kernel[0], 1, sizeof(cl_mem), &stn_conv1_wt_buf);
    checkError(status, "Failed to set argument 1 for stn_conv1 ");


    status = clSetKernelArg(kernel[0],2, sizeof(cl_mem), &stn_conv1_bias_buf);
    checkError(status, "Failed to set argument 2 for stn_conv1");


    status = clSetKernelArg(kernel[0],3, sizeof(cl_mem), &inout_buf);
    checkError(status, "Failed to set argument 3 for stn_conv1 ");


    const size_t global_work_size = 1;    //single work item kernel

    printf("Launching for device kernel stn_conv1 (%zd elements)\n", global_work_size);


    //1D NDRange (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[0], 1, NULL,
            &global_work_size, NULL,0, NULL, &kernel_event[0]);
    checkError(status, "Failed to launch kernel for stn_conv1");


   // clFinish(queue); 			//added


    //stn_conv2
    status = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), &inout_buf);
    checkError(status, "Failed to set argument 0 of stn_conv2");


    status = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), &stn_conv2_wt_buf);
    checkError(status, "Failed to set argument 1 of stn_conv2");


    status = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), &stn_conv2_bias_buf);
    checkError(status, "Failed to set argument 2 of stn_conv2");


    status = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), &inout1_buf);
    checkError(status, "Failed to set argument 3 of stn_conv2 ");


    printf("Launching for device kernel stn_conv2 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[0]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[1], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[0],&kernel_event[1]);
    checkError(status, "Failed to launch kernel for stn_conv2");

   // clFinish(queue); 			//added

    //FC1

    status = clSetKernelArg(kernel[2], 0, sizeof(cl_mem), &inout1_buf);
    checkError(status, "Failed to set argument 0 of FC1");


    status = clSetKernelArg(kernel[2], 1, sizeof(cl_mem), &fc1_wt_buf);
    checkError(status, "Failed to set argument 1 of FC1");


    status = clSetKernelArg(kernel[2], 2, sizeof(cl_mem), &fc1_bias_buf);
    checkError(status, "Failed to set argument 2 of FC1");


    status = clSetKernelArg(kernel[2], 3, sizeof(cl_mem), &inout2_buf);
    checkError(status, "Failed to set argument 3 of FC1 ");

    const size_t global_FC1_work_size[2] = {1,100} ; 
    printf("Launching FC1 for device  (global size: %zd, %zd)\n", global_FC1_work_size[0], global_FC1_work_size[1]);

    clWaitForEvents(1,&kernel_event[1]);   //added

    //2D ND Range 
    status = clEnqueueNDRangeKernel(queue, kernel[2], 2, NULL,
            global_FC1_work_size, NULL,1,&kernel_event[1],&kernel_event[2]);
    checkError(status, "Failed to launch kernel for FC1"); 

   // clFinish(queue); 			//added

    //FC2

    status = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &inout2_buf);
    checkError(status, "Failed to set argument 0 of FC2");


    status = clSetKernelArg(kernel[3], 1, sizeof(cl_mem), &fc2_wt_buf);
    checkError(status, "Failed to set argument 1 of FC2");


    status = clSetKernelArg(kernel[3], 2, sizeof(cl_mem), &fc2_bias_buf);
    checkError(status, "Failed to set argument 2 of FC2");

    status = clSetKernelArg(kernel[3], 3, sizeof(cl_mem), &theta_buf);
    checkError(status, "Failed to set argument 3 of FC2 ");

    // const size_t global_FC2_work_size[2] = {1,6} ; 
    //  printf("Launching FC2 for device  (global size: %zd, %zd)\n", global_FC2_work_size[0], global_FC2_work_size[1]);


    clWaitForEvents(1,&kernel_event[2]);   //added

    printf("Launching for device kernel FC2 (%zd elements)\n", global_work_size);

    //2D ND Range 

    status = clEnqueueNDRangeKernel(queue, kernel[3], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[2],&kernel_event[3]);
    checkError(status, "Failed to launch kernel for FC2");


    //   clWaitForEvents(1,&kernel_event[3]);   //added

   // clFinish(queue); 			//added

    //stn_func

    status = clSetKernelArg(kernel[4], 0, sizeof(cl_mem), &image_buf);
    checkError(status, "Failed to set argument 0 of stn_func");


    status = clSetKernelArg(kernel[4], 1, sizeof(cl_mem), &theta_buf);
    checkError(status, "Failed to set argument 1 of stn_func");


    status = clSetKernelArg(kernel[4], 2, sizeof(cl_mem), &inout3_buf);
    checkError(status, "Failed to set argument 2 of stn_func");


    clWaitForEvents(1,&kernel_event[3]);   //added

    printf("Launching for device kernel stn_func (%zd elements)\n", global_work_size);


    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[4], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[3],&kernel_event[4]);
    checkError(status, "Failed to launch kernel for stn_func");


   // clFinish(queue); 			//added

    //conv1

    //  status = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &image_buf);
    status = clSetKernelArg(kernel[5], 0, sizeof(cl_mem), &inout3_buf);
    checkError(status, "Failed to set argument 0 of conv1");


    status = clSetKernelArg(kernel[5], 1, sizeof(cl_mem), &conv1_wt_buf);
    checkError(status, "Failed to set argument 1 of conv1");


    status = clSetKernelArg(kernel[5], 2, sizeof(cl_mem), &conv1_bias_buf);
    checkError(status, "Failed to set argument 2 of conv1");


    status = clSetKernelArg(kernel[5], 3, sizeof(cl_mem), &inout4_buf);
    checkError(status, "Failed to set argument 4 of conv1 ");

    printf("Launching for device kernel conv1 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[4]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[5], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[4],&kernel_event[5]);
    checkError(status, "Failed to launch kernel for conv1");


   // clFinish(queue); 			//added


    //bn1

    status = clSetKernelArg(kernel[6], 0, sizeof(cl_mem), &inout4_buf);
    checkError(status, "Failed to set argument 0 of bn1 ");

    status = clSetKernelArg(kernel[6], 1, sizeof(cl_mem), &bn1_mean_buf);
    checkError(status, "Failed to set argument 1 of bn1 ");

    status = clSetKernelArg(kernel[6], 2, sizeof(cl_mem), &bn1_variance_buf);
    checkError(status, "Failed to set argument 2 of bn1 ");

    status = clSetKernelArg(kernel[6], 3, sizeof(cl_mem), &bn1_gamma_buf);
    checkError(status, "Failed to set argument 3 of bn1 ");


    status = clSetKernelArg(kernel[6], 4, sizeof(cl_mem), &bn1_beta_buf);
    checkError(status, "Failed to set argument 4 of bn1 ");


    status = clSetKernelArg(kernel[6], 5, sizeof(cl_mem), &inout5_buf);
    checkError(status, "Failed to set argument 5 of bn1 ");


    printf("Launching for device kernel bn1 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[5]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[6], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[5],&kernel_event[6]);
    checkError(status, "Failed to launch kernel for bn1");



    //clFinish(queue); 			//added


    //conv2

    status = clSetKernelArg(kernel[7], 0, sizeof(cl_mem), &inout5_buf);
    checkError(status, "Failed to set argument 0 of conv2");


    status = clSetKernelArg(kernel[7], 1, sizeof(cl_mem), &conv2_wt_buf);
    checkError(status, "Failed to set argument 1 of conv2");


    status = clSetKernelArg(kernel[7], 2, sizeof(cl_mem), &conv2_bias_buf);
    checkError(status, "Failed to set argument 2 of conv2");


    status = clSetKernelArg(kernel[7], 3, sizeof(cl_mem), &inout6_buf);
    checkError(status, "Failed to set argument 3 of conv1 ");


    printf("Launching for device kernel conv2 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[6]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[7], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[6],&kernel_event[7]);
    checkError(status, "Failed to launch kernel for conv2");

    //clFinish(queue); 			//added


    //bn2

    status = clSetKernelArg(kernel[8], 0, sizeof(cl_mem), &inout6_buf);
    checkError(status, "Failed to set argument 0 of bn2 ");

    status = clSetKernelArg(kernel[8], 1, sizeof(cl_mem), &bn2_mean_buf);
    checkError(status, "Failed to set argument 1 of bn2 ");

    status = clSetKernelArg(kernel[8], 2, sizeof(cl_mem), &bn2_variance_buf);
    checkError(status, "Failed to set argument 2 of bn2 ");

    status = clSetKernelArg(kernel[8], 3, sizeof(cl_mem), &bn2_gamma_buf);
    checkError(status, "Failed to set argument 3 of bn2 ");


    status = clSetKernelArg(kernel[8], 4, sizeof(cl_mem), &bn2_beta_buf);
    checkError(status, "Failed to set argument 4 of bn2 ");


    status = clSetKernelArg(kernel[8], 5, sizeof(cl_mem), &inout7_buf);
    checkError(status, "Failed to set argument 5 of bn2 ");


    printf("Launching for device kernel bn2 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[7]);   //added


    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[8], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[7],&kernel_event[8]);
    checkError(status, "Failed to launch kernel for bn2");


    //clFinish(queue); 			//added


    //conv3

    status = clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &inout7_buf);
    checkError(status, "Failed to set argument 0 of conv3");


    status = clSetKernelArg(kernel[9], 1, sizeof(cl_mem), &conv3_wt_buf);
    checkError(status, "Failed to set argument 1 of conv3");


    status = clSetKernelArg(kernel[9], 2, sizeof(cl_mem), &conv3_bias_buf);
    checkError(status, "Failed to set argument 2 of conv3");


    status = clSetKernelArg(kernel[9], 3, sizeof(cl_mem), &inout8_buf);
    checkError(status, "Failed to set argument 3 of conv3 ");

    printf("Launching for device kernel conv3 (%zd elements)\n", global_work_size);


    clWaitForEvents(1,&kernel_event[8]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[9], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[8],&kernel_event[9]);
    checkError(status, "Failed to launch kernel for conv3");


   // clFinish(queue); 			//added


    //bn3	

    status = clSetKernelArg(kernel[10], 0, sizeof(cl_mem), &inout8_buf);
    checkError(status, "Failed to set argument 0 of bn3 ");

    status = clSetKernelArg(kernel[10], 1, sizeof(cl_mem), &bn3_mean_buf);
    checkError(status, "Failed to set argument 1 of bn3 ");

    status = clSetKernelArg(kernel[10], 2, sizeof(cl_mem), &bn3_variance_buf);
    checkError(status, "Failed to set argument 2 of bn3 ");

    status = clSetKernelArg(kernel[10], 3, sizeof(cl_mem), &bn3_gamma_buf);
    checkError(status, "Failed to set argument 3 of bn3 ");


    status = clSetKernelArg(kernel[10], 4, sizeof(cl_mem), &bn3_beta_buf);
    checkError(status, "Failed to set argument 4 of bn3 ");


    status = clSetKernelArg(kernel[10], 5, sizeof(cl_mem), &inout9_buf);
    checkError(status, "Failed to set argument 5 of bn3 ");

    clWaitForEvents(1,&kernel_event[9]);   //added

    printf("Launching for device kernel bn3 (%zd elements)\n", global_work_size);


    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[10], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[9],&kernel_event[10]);
    checkError(status, "Failed to launch kernel for bn3"); 


   // clFinish(queue); 			//added


    //FC3

    status = clSetKernelArg(kernel[11], 0, sizeof(cl_mem), &inout9_buf);
    checkError(status, "Failed to set argument 0 of FC3");


    status = clSetKernelArg(kernel[11], 1, sizeof(cl_mem), &fc3_wt_buf);
    checkError(status, "Failed to set argument 1 of FC3");


    status = clSetKernelArg(kernel[11], 2, sizeof(cl_mem), &fc3_bias_buf);
    checkError(status, "Failed to set argument 2 of FC3");


    status = clSetKernelArg(kernel[11], 3, sizeof(cl_mem), &inout10_buf);
    checkError(status, "Failed to set argument 3 of FC3 ");


   // printf("Launching for device kernel FC3 (%zd elements)\n", global_work_size);


    //    cl_event fc3_event_waitlist[3] ;
    //    fc3_event_waitlist[0] = write_event[0];
    //    fc3_event_waitlist[1] = write_event[1];
    //    fc3_event_waitlist[2] = kernel_event[10];


    clWaitForEvents(1,&kernel_event[10]);   //added

    //1D ND Range (1,1,1)
 /*   status = clEnqueueNDRangeKernel(queue, kernel[11], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[10],&kernel_event[11]);
    checkError(status, "Failed to launch kernel for FC3"); */

    const size_t global_FC3_work_size[2] = {1,350} ; 

    printf("Launching FC3 for device  (global size: %zd, %zd)\n", global_FC3_work_size[0], global_FC1_work_size[3]);
   
   
   //2D ND Range 
    status = clEnqueueNDRangeKernel(queue, kernel[11], 2, NULL,
            global_FC3_work_size, NULL,1,&kernel_event[10],&kernel_event[11]);
    checkError(status, "Failed to launch kernel for FC3"); 


    //clFinish(queue); 			//added

    //bn4	

    status = clSetKernelArg(kernel[12], 0, sizeof(cl_mem), &inout10_buf);
    checkError(status, "Failed to set argument 0 of bn4 ");

    status = clSetKernelArg(kernel[12], 1, sizeof(cl_mem), &bn4_mean_buf);
    checkError(status, "Failed to set argument 1 of bn4 ");

    status = clSetKernelArg(kernel[12], 2, sizeof(cl_mem), &bn4_variance_buf);
    checkError(status, "Failed to set argument 2 of bn4 ");

    status = clSetKernelArg(kernel[12], 3, sizeof(cl_mem), &bn4_gamma_buf);
    checkError(status, "Failed to set argument 3 of bn4 ");


    status = clSetKernelArg(kernel[12], 4, sizeof(cl_mem), &bn4_beta_buf);
    checkError(status, "Failed to set argument 4 of bn4 ");


    status = clSetKernelArg(kernel[12], 5, sizeof(cl_mem), &inout11_buf);
    checkError(status, "Failed to set argument 5 of bn4 ");


    clWaitForEvents(1,&kernel_event[11]);   //added

    printf("Launching for device kernel bn4 (%zd elements)\n", global_work_size);


    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[12], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[11],&kernel_event[12]);
    checkError(status, "Failed to launch kernel for bn4"); 


   // clFinish(queue); 			//added


    //FC4

    status = clSetKernelArg(kernel[13], 0, sizeof(cl_mem), &inout11_buf);
    checkError(status, "Failed to set argument 0 of FC4");


    status = clSetKernelArg(kernel[13], 1, sizeof(cl_mem), &fc4_wt_buf);
    checkError(status, "Failed to set argument 1 of FC4");


    status = clSetKernelArg(kernel[13], 2, sizeof(cl_mem), &fc4_bias_buf);
    checkError(status, "Failed to set argument 2 of FC4");


    status = clSetKernelArg(kernel[13], 3, sizeof(cl_mem), &output_buf);
    checkError(status, "Failed to set argument 3 of FC4 ");


    printf("Launching for device kernel FC4 (%zd elements)\n", global_work_size);

    //    cl_event fc3_event_waitlist[3] ;
    //    fc3_event_waitlist[0] = write_event[0];
    //    fc3_event_waitlist[1] = write_event[1];
    //    fc3_event_waitlist[2] = kernel_event[10];


    clWaitForEvents(1,&kernel_event[12]);   //added

    //1D ND Range (1,1,1)
    status = clEnqueueNDRangeKernel(queue, kernel[13], 1, NULL,
            &global_work_size, NULL,1,&kernel_event[12],&kernel_event[13]);
    checkError(status, "Failed to launch kernel for FC4"); 
   // clFinish(queue); 			//added


    clWaitForEvents(1,&kernel_event[13]);   //added


    //Reading output 


    //   clWaitForEvents(1,&kernel_event[11]);   //added

    //  const double end_time = getCurrentTimestamp();

    /*    // Wall-clock time taken.
          printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

    // Get kernel times using the OpenCL event profiling API.
    cl_ulong time_ns = getStartEndTime(kernel_event[1]);
    printf("Kernel time (device ): %0.3f ms\n", double(time_ns) * 1e-6); */


    status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE,0, N24* sizeof(float), debug_output,1,&kernel_event[13], NULL);

    //    clReleaseEvent(write_event[0]);
    //    clReleaseEvent(write_event[1]);


    // Release all events.
    clReleaseEvent(kernel_event[0]);
    clReleaseEvent(kernel_event[1]);
    clReleaseEvent(kernel_event[2]);
    clReleaseEvent(kernel_event[3]);
    clReleaseEvent(kernel_event[4]);
    clReleaseEvent(kernel_event[5]);
    clReleaseEvent(kernel_event[6]);
    clReleaseEvent(kernel_event[7]);
    clReleaseEvent(kernel_event[8]);
    clReleaseEvent(kernel_event[9]);
    clReleaseEvent(kernel_event[10]);
    clReleaseEvent(kernel_event[11]);
    clReleaseEvent(kernel_event[12]);
    clReleaseEvent(kernel_event[13]);



    //Printing the results of output

    for ( unsigned j = 0 ; j < N24 ; j ++) {
        //   printf ("\nContents of debug_output[%d] = %f",j,debug_output[j]);
        printf ("%f,",debug_output[j]);
    }

}

// Free the resources allocated during initialization
void cleanup() {

    clReleaseKernel(kernel[0]);
    clReleaseKernel(kernel[1]);
    clReleaseKernel(kernel[2]);
    clReleaseKernel(kernel[3]);
    clReleaseKernel(kernel[4]);
    clReleaseKernel(kernel[5]);
    clReleaseKernel(kernel[6]);
    clReleaseKernel(kernel[7]);
    clReleaseKernel(kernel[8]);
    clReleaseKernel(kernel[9]);
    clReleaseKernel(kernel[10]);
    clReleaseKernel(kernel[11]);
    clReleaseKernel(kernel[12]);
    clReleaseKernel(kernel[13]);



    clReleaseCommandQueue(queue);



    clReleaseMemObject(image_buf);
    clReleaseMemObject(stn_conv1_wt_buf);
    clReleaseMemObject(stn_conv1_bias_buf);

    clReleaseMemObject(inout_buf);
    clReleaseMemObject(stn_conv2_wt_buf);
    clReleaseMemObject(stn_conv2_bias_buf);

    clReleaseMemObject(inout1_buf);
    clReleaseMemObject(fc1_wt_buf);
    clReleaseMemObject(fc1_bias_buf);
    clReleaseMemObject(inout2_buf);


    clReleaseMemObject(fc2_wt_buf);
    clReleaseMemObject(fc2_bias_buf);
    clReleaseMemObject(theta_buf);


    clReleaseMemObject(inout3_buf);
    clReleaseMemObject(conv1_wt_buf);
    clReleaseMemObject(conv1_bias_buf);
    clReleaseMemObject(inout4_buf);

    clReleaseMemObject(bn1_mean_buf);
    clReleaseMemObject(bn1_variance_buf);
    clReleaseMemObject(bn1_gamma_buf);
    clReleaseMemObject(bn1_beta_buf);
    clReleaseMemObject(inout5_buf);

    clReleaseMemObject(conv2_wt_buf);
    clReleaseMemObject(conv2_bias_buf);
    clReleaseMemObject(inout6_buf);

    clReleaseMemObject(bn2_mean_buf);
    clReleaseMemObject(bn2_variance_buf);
    clReleaseMemObject(bn2_gamma_buf);
    clReleaseMemObject(bn2_beta_buf);
    clReleaseMemObject(inout7_buf);

    clReleaseMemObject(conv3_wt_buf);
    clReleaseMemObject(conv3_bias_buf);
    clReleaseMemObject(inout8_buf);

    clReleaseMemObject(bn3_mean_buf);
    clReleaseMemObject(bn3_variance_buf);
    clReleaseMemObject(bn3_gamma_buf);
    clReleaseMemObject(bn3_beta_buf);
    clReleaseMemObject(inout9_buf);


    clReleaseMemObject(fc3_wt_buf);
    clReleaseMemObject(fc3_bias_buf);
    clReleaseMemObject(inout10_buf);


    clReleaseMemObject(bn4_mean_buf);
    clReleaseMemObject(bn4_variance_buf);
    clReleaseMemObject(bn4_gamma_buf);
    clReleaseMemObject(bn4_beta_buf);
    clReleaseMemObject(inout11_buf);


    clReleaseMemObject(fc4_wt_buf);
    clReleaseMemObject(fc4_bias_buf);
    clReleaseMemObject(output_buf);



    clReleaseProgram(program);
    clReleaseContext(context);




    free(source_image);
    free(stn_conv1_wt);
    free(stn_conv1_bias);

    free(stn_conv2_wt);
    free(stn_conv2_bias);

    free(fc1_wt);
    free(fc1_bias);

    free(fc2_wt);
    free(fc2_bias);

    free(conv1_wt);
    free(conv1_bias);

    free(bn1_mean);
    free(bn1_variance);
    free(bn1_gamma);
    free(bn1_beta);

    free(conv2_wt);
    free(conv2_bias);

    free(bn2_mean);
    free(bn2_variance);
    free(bn2_gamma);
    free(bn2_beta);

    free(conv3_wt);
    free(conv3_bias);


    free(bn3_mean);
    free(bn3_variance);
    free(bn3_gamma);
    free(bn3_beta);

    free(fc3_wt);
    free(fc3_bias);

    free(bn4_mean);
    free(bn4_variance);
    free(bn4_gamma);
    free(bn4_beta);

    free(fc4_wt);
    free(fc4_bias);

    //    free(output);
    free(debug_output);
}

