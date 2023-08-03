# Randomize Text
- cat file.txt | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .2) print $0}' > train_r.txt

# Change Encoding
- vim +"set nobomb | set fenc=utf8 | x" out.txt

# Create Soft Link
- sudo ln -s /usr/local/cuda-10.0/lib64/libcusparse.so.10.0 /usr/lib/x86_64-linux-gnu/libcusparse.so.10

# Splitting and Shuffling
- cat file1.txt file2.txt file3.txt | shuf > all.txt
- split -l $[ $(wc -l all.txt|cut -d" " -f1) * 80/ 100 ] all.txt
- mv xaa Train.txt
- mv xab Test.txt