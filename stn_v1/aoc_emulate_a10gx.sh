#!/bin/bash
#Compiler optimization flags used
#-no-interleaving=default -fp-relaxed -fpc 
aoc -march=emulator   device/traffic_stn.cl  -o bin/traffic_stn.aocx  -board=a10gx

