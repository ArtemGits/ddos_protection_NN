#!/bin/bash
output_dir=$1
inotifywait -m "$output_dir" -e create -e moved_to |
    while read output_dir action file; do
        if [[ "$file" =~ .*csv ]]; then
            echo "csv file "$file"" # If so, do your thing here!
	    python3 python_loader_to_kafka.py $file 
        fi
    done
