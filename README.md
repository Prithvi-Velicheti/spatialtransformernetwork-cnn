# spatialtransformernetwork-cnn

 Traffic Sign Classification using Spatial Transformer Networks.![hardware_arch](https://user-images.githubusercontent.com/25413124/142775588-e9ac41ac-349a-48d9-a9f9-2e72c1a059bc.png)

 
 Requirements:     \
 Install INTEL FPGA SDK FOR OPENCL   \
 
 To Emulate and Run on CPU   <br/>
 $ ./aoc_emulate_a10gx.sh    <br/>
 $  make  <br/>
 $  CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host <img_num>    \
 <img_num> ranges from 1-9   \
 
 To Run on Hardware  \
 $./aoc_compile_a10gx.sh   { compilation takes 2 hours approx }   \
 $ aocl program acl0 traffic_stn.aocx    \
 $ bin/host   \
 
 
 
