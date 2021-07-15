#!/bin/bash

SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

for f in ./*/labels/*.txt
  do  
    # sed -i 's/^4/0/; s/^2/x/; s/^3/2/; s/^11/3/; s/^1/4/; s/^x/1/' $f
    # sed -i 's/^7/1/; s/^8/4/; s/^12/2/' $f
    sed -i '/^3/d; s/^4/3/' $f
  done

IFS=$SAVEIFS
