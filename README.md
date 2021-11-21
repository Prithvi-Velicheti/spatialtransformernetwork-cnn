# spatialtransformernetwork-cnn
![hardware_arch](https://user-images.githubusercontent.com/25413124/142775291-e718ad72-7cb3-4d2d-811a-598ab8dbb5fb.png=250,250)
 Traffic Sign Classification using Spatial Transformer Networks.
 
 Requirements:
 Install INTEL FPGA SDK FOR OPENCL 
 
 To Emulate and Run on CPU
 $ ./aoc_emulate_a10gx.sh
 $  make
 $  CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host <img_num> 
 <img_num> ranges from 1-9
 
 To Run on Hardware
 $./aoc_compile_a10gx.sh   { compilation takes 2 hours approx }
 $ aocl program acl0 traffic_stn.aocx 
 $ bin/host
 
 
 
