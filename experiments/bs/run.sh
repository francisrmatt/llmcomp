#!/bin/sh
xpd=$(pwd)
cd ../..
export LLMCOMP_LOG_LEVEL=INFO
python go.py --action=train --compressor=btransformer --which=bs_01 --amt=5000000 --shh >> $xpd/logn.out 2>&1
python go.py --action=train --compressor=btransformer --which=bs_02 --amt=1000000 --shh >> $xpd/logn.out 2>&1
python go.py --action=train --compressor=btransformer --which=bs_03 --amt=500000 --shh >> $xpd/logn.out 2>&1
python go.py --action=train --compressor=btransformer --which=bs_04 --amt=250000 --shh >> $xpd/logn.out 2>&1
python go.py --action=train --compressor=btransformer --which=bs_05 --amt=100000 --shh >> $xpd/logn.out 2>&1










