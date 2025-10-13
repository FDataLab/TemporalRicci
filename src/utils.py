import csv

def txt_to_csv_with_headers(txt_file_path, csv_file_path, default_value=1):
    with open(txt_file_path, 'r') as txt_file:
        lines = txt_file.readlines()

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['from', 'to', 'value', 'timestamp'])  # header

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                from_node = parts[0]
                to_node = parts[1]
                timestamp = parts[2]
                writer.writerow([from_node, to_node, default_value, timestamp])

txt_to_csv_with_headers('./data/mathoverflow.txt', './data/mathoverflow.csv')
