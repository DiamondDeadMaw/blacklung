import lzma
import shutil
import os
from tqdm import tqdm


def compress_file(input_file, output_file):
    try:
        print(f"Starting compression of {input_file} to {output_file}...")
        total_size = os.path.getsize(input_file)

        with open(input_file, 'rb') as f_in:
            with lzma.open(output_file, 'wb') as f_out:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Compressing") as pbar:
                    buffer_size = 1024 * 1024
                    while chunk := f_in.read(buffer_size):
                        f_out.write(chunk)
                        pbar.update(len(chunk))

        print(f"Compressed file successfully: {output_file}")
    except Exception as e:
        print(f"Error compressing file: {e}")


def decompress_file(input_file, output_file):
    try:
        print(f"Starting decompression of {input_file} to {output_file}...")
        total_size = os.path.getsize(input_file)

        with lzma.open(input_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Decompressing") as pbar:
                    buffer_size = 1024 * 1024
                    while chunk := f_in.read(buffer_size):
                        f_out.write(chunk)
                        pbar.update(len(chunk))

        print(f"Decompressed file successfully: {output_file}")
    except Exception as e:
        print(f"Error decompressing file: {e}")


input_file = 'FinalData10Stations.csv'
output_file = 'FinalData10Stations.csv.xz'

# Set this to True if you want to decompress, or False to compress
decompress = False

if not os.path.isfile(input_file):
    print(f"Error: The file {input_file} does not exist.")
else:
    print(f"Found input file: {input_file}")
    if decompress:
        if not input_file.endswith('.xz'):
            print("Error: For decompression, the input file must have an '.xz' extension.")
        else:
            print(f"Starting decompression process...")
            decompress_file(input_file, output_file)
    else:
        if not input_file.endswith('.csv'):
            print("Error: For compression, the input file must have a '.csv' extension.")
        else:
            print(f"Starting compression process...")
            compress_file(input_file, output_file)
