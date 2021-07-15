#!/bin/bash

i=1
for f in ./images/*.jpg ; do
	i=`expr $i '+' 1`
	if [ $i -le 6007 ] ; then
		mv "$f" ./train/images
	else
		mv "$f" ./valid/images
	fi
done

#j=1
#for f in ./labels/*.txt ; do
#        j=`expr $j '+' 1`
#        if [ $j -le 6007 ] ; then
#                mv "$f" ./train/labels
#        else
#                mv "$f" ./valid/labels
#        fi
#done
