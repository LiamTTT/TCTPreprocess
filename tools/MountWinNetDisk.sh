#!/bin/bash
echo "Shell Mount Net Disk of Windows OS."
echo "URL of disk: $1"
echo "Location of mounted dir: $2"
echo "Username: $3"
#echo "Password: $4"

if [ ! -d $2 ]; then
  echo "create $2"
  mkdir $2
fi

sudo mount -t cifs -o username=$3,password=$4 $1 $2

echo "done!"
