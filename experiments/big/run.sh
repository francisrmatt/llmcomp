#!/bin/sh
xpd=$(pwd)
cd ../..

python go.py --action=train --compressor=btransformer --which=c2048_test --amt=150000 >> $xpd/log.out 2>&1








