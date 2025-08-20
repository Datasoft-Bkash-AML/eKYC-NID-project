#!/bin/bash
# Delete all .pyc files in the repository

find . -name "*.pyc" -type f -delete
echo "All .pyc files deleted."
