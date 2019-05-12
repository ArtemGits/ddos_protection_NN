#!/bin/bash
filename='black_list.txt'
monitoring_dir=$1

inotifywait -m "$monitoring_dir" -e create -e moved_to | # what is it?
    while read monitoring_dir action file; do
        if [[ "$file" =~ $filename ]]; then
            echo "add ip addres from black list to iptables rule" 
	    while read ip; do
            #iptables -D INPUT -s $ip -j DROP
            iptables -I INPUT -s $ip -j DROP #-A
        done < $1/$filename
        rm $1/$filename
        invoke-rc.d iptables-persistent save
        fi
    done

