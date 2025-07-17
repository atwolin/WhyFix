#!/usr/bin/env bash
# coding=utf-8

# This script is used to parse the pickled data from Cambridge Dictionary

for i in {a..z}; do
    if [ "$i" == "z" ]; then
        next="aa"
    else
        next=$(echo $i | tr "a-y" "b-z")
    fi
    echo "Parsing data from $i to $next"
    python parser.py $i $next
done

# echo "Parsing data"
# python parser.py