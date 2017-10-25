#!/bin/bash
for i in `seq 1 60`;
do
  echo $i
  python3 test.py
done
