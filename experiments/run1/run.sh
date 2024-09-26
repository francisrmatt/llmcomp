#!/bin/sh
xpd=$(pwd)
cd ../..
python go.py --action=train --compressor=btransformer --which=c512_002 --amt=100 --shh >> $xpd/log.out 2>&1
python go.py --action=compress --compressor=btransformer --which=c512_002 --amt=100 --file=-2 >> $xpd/log.out 2>&1
