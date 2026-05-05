#!/bin/bash

rsync -avhP \
  --files-from=target_weights.txt \
  bacobax2@killarney.alliancecan.ca:/home/bacobax2/projects/aip-mpederso/bacobax2/ir-generative-study/ \
  /Users/francescobassignana/Desktop/school/unitn/internship/canada/IR-generative-study && rsync -avhP \
  --files-from=target_weights.txt \
  /Users/francescobassignana/Desktop/school/unitn/internship/canada/IR-generative-study/ \
  Fbassignana@bool4.livia.etsmtl.ca:/projets/Fbassignana/diffusers_try/flow_matching_trial
