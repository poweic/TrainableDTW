#!/bin/bash -xe
./train -p train --mfcc-root=/share/mlp_posterior/gaussian_posterior_noprior_no_log/ --model=diag --batch-size=256 --feat-dim=76 --learning-rate=0.5 --theta-output=exp/theta/theta.rand.1
