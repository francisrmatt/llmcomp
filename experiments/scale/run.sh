#!/bin/sh
xpd=$(pwd)
cd ../..
export LLMCOMP_LOG_LEVEL=DEBUG
python go.py --action=train --compressor=btransformer --which=scale_01 --amt=2500000 >> $xpd/log.out 2>&1