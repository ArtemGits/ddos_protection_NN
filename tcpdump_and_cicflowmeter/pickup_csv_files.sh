#!/bin/bash
output_dir=$1
inotifywait -m "$output_dir" -e create -e moved_to |
    while read output_dir action file; do
        if [[ "$file" =~ .*csv ]]; then
            echo "csv file "$file"" # If so, do your thing here!
      path=`pwd`/$output_dir$file
      echo $path
	    python3 ../python-client/python_loader_to_kafka.py $path 
        fi
    done
