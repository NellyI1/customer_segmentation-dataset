import pandas as pd
import random
import csv

input_file = '/Users/ifeomaigbokwe/Downloads/amz_us_price_prediction_dataset.csv'
output_file = 'random_sample_15000_records.csv'
sample_size = 15000

# Open the file using CSV reader for line-by-line reading
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read header separately
    
    reservoir = []
    for i, row in enumerate(reader):
        if i < sample_size:
            reservoir.append(row)
        else:
            # Randomly replace elements in the reservoir with decreasing probability
            j = random.randint(0, i)
            if j < sample_size:
                reservoir[j] = row

# Write sampled data to a new CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)  # Write header
    writer.writerows(reservoir)

print(f"Random sample of {sample_size} records saved to '{output_file}'")
