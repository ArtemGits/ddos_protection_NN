import csv
fname_in = 'Wednesday-workingHours.pcap_ISCX.csv' 
fname_out = 'Wednesday-workingHours.pcap_ISCX_out.csv' 
inc_f = open(fname_in,'r')  #open the file for reading
csv_r = csv.reader(inc_f) # Attach the csv "lens" to the input stream - default is excel dialect
out_f = open(fname_out,'w') #open the file for writing
csv_w = csv.writer(out_f, delimiter=',',lineterminator='\n' ) #attach the csv "lens" to the stream headed to the output file
for row in csv_r: #Loop Through each row in the input file
    new_row = row[:]  # initialize the output row
    new_row.pop(61) #Whatever column you wanted to delete
    csv_w.writerow(new_row) 
inc_f.close()
out_f.close()
