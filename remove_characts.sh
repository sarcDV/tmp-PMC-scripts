#!/bin/bash

cd AES_analysis/
# cd CustomGradientEntropy_analysis/

for file in *.txt;
do 
	echo "$file"
	new_name_=$(echo "$file" | cut -f 1 -d '.')
	# echo "$new_name_"-test.txt
	sed -i s/,//g "$file"
	sed -i s/\(//g "$file"
	sed -i s/\)//g "$file"
	sed -i s/\'//g "$file"
	sed -i 's/\\n//g' "$file"
	
done

# sed -e 's/.*]//' -e 's/  */ /g' file.txt
# sed 's/.*]//' file.txt

# tr can do that:
#tr -d \" < infile > outfile
#You could also use sed:
#sed 's/"//g' < infile > outfile
# sed -i s/\"//g file.txt
# Use single quotes for the expression you used:

#sed 's/\//\\\//g'

#In double quotes, \ has a special meaning, so you have to backslash it:

## sed "s/\//\\\\\//g"
# But it's cleaner to change the delimiter:

#sed 's=/=\\/=g'
# sed "s=/=\\\/=g"

# sed 's/(//g' < "$file" > "$new_name_"-test.txt
# sed 's/\n//' < "$new_name_"-test.txt > "$new_name_"-test2.txt
# sed 's/\n//' "$file" > "$new_name_"-test.txt
# sed 's/,//' "$new_name_"-test.txt > "$new_name_"-test2.txt
# sed 's///' "$new_name_"-test2.txt > "$new_name_"-test3.txt