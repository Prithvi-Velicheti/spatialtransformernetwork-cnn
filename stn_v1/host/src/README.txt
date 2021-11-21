
kernel stn_conv1  N1 = 1024 , N2 = 2450 , N3 = 50 , N4 = 8450

Host Mem : source_image (32 *32), stn_conv1_wt(7*7*50) ,stn_conv1_bias(50) 

Device Mem : image_buf, stn_conv1_wt_buf, stn_conv1_bias_buf , inout_buf [13*13*50] 

kernel stn_conv2  N5=125000 , N6 = 100 ,N7=1600

Host Mem  stn_conv2_wt(5*5*50*100)  stn_conv2_bias(100) 

Device Mem   inout_buf, stn_conv2_wt_buf, stn_conv2_bias_buf , inout1_buf[4*4*100]


FC1         N7, N6 ,N8 = 1600*100
Host Mem :  fc1_wt (1600*100) , fc1_bias(100)

Device Mem : inout1_buf , fc1_wt_buf, fc1_bias_buf , inout2_buf[100]

FC2         N9 = 600 , N10 = 6
Host Mem : fc2_wt(6*100) fc2_bias(6*1)

Device Mem : inout2_buf, fc2_wt_buf, fc2_bias_buf , theta_buf[6]

STN_func  N1, N10 ,N1

Host Mem : -
Device Mem : image_buf ,theta_buf, inout3_buf 

conv1 : N11 = 2500 , N6 = 100 , N12 = 14*14*100

Host Mem : conv1_wt[100*5*5],conv1_bias[100]
Device mem : inout3_buf , conv1_wt_buf , conv1_bias_buf , inout4_buf[14*14*100]

bn1 : N6 , N12 

Host Mem : bn1_mean[100], bn1_variance[100],bn1_gamma[100], bn1_beta[100] 
Device Mem : inout4_buf ,bn1_mean_buf, bn1_variance_buf, bn1_gamma_buf, bn1_beta_buf, inout5_buf



conv2  N13 = 150*100*3*3 , N14 = 150, N15 = 6*6*150

Host mem   conv2_wt[150*100*3*3] , conv2_bias[150];
Device Mem inout5_buf,conv2_wt_buf,conv2_bias_buf , inout6_buf[6*6*150]

bn2  N14 ,N15
Host mem   bn2_mean[150], bn2_variance[150] ,bn2_gamma[150], bn2_beta[150] ;
Device mem inout6_buf , bn2_mean_buf ,bn2_variance_buf, bn2_gamma_buf, bn2_beta_buf, inout7_buf



conv3   N16 = 250*150 , N17 = 250 , N18 = 3*3*250 
Host Mem   conv3_wt[250*150] conv3_bias[250] 

Device mem inout7_buf, conv3_wt_buf , conv3_bias_buf , inout8_buf[3*3*250]


bn3   N17 , N18
Host mem  bn3_mean[250], bn3_variance[250] ,bn3_gamma[250], bn3_beta[250]
Device mem inout8_buf , bn3_mean_buf ,bn3_variance_buf, bn3_gamma_buf, bn3_beta_buf, inout9_buf


FC3   N19 = 250*3*3*350 N20 = 350 
Host Mem fc3_wt[250*3*3*350] , fc3_bias[350]
Device mem inout9_buf[3*3*250] ,fc3_wt_buf , fc3_bias_buf, inout10_buf[350]



bn4 N20 , 

Host mem  bn4_mean[350], bn4_variance[350] ,bn4_gamma[350], bn4_beta[350]
Device mem inout10_buf , bn4_mean_buf ,bn4_variance_buf, bn4_gamma_buf, bn4_beta_buf, inout11_buf[350]


fc4 (softmax)  N21 = 350*43 , N22 = 43 , N23 = 1 ;
Host mem  fc4_wt[350*43] fc4_bias[43] , output[1]

Device mem : inout11_buf fc4_wt_buf, fc4_bias_buf, output_buf[1] 























