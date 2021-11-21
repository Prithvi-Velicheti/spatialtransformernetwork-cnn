#!/bin/bash
#Compiler optimization flags used
#-no-interleaving=default -fp-relaxed -fpc 
aoc -march=emulator device/*.cl  -o bin/traffic_stn.aocx  -board=s5_ref

