import csv
import math

dev_data_dict = []
# https://docs.python.org/3/library/csv.html
csv.field_size_limit(991072)
with open('./full_output.csv', newline='', encoding="utf8") as csvfile:
    dev_data_csv = csv.reader(csvfile, delimiter=',')
    for row in dev_data_csv:
        if row[0] != "Query_number":
            if len(row) != 5:
                print("error in input")
                print(row)
            else:
                dev_data_dict.append(row)

dev_data_dict.sort(key=lambda x: (x[0], -float(x[4])))
dev_data_dict = [[row[0], row[1], row[2], row[3], str(math.floor(float(row[4]) + 0.5))] for row in dev_data_dict]

with open('./neural_model_rank.csv', 'w+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    header = ["Query_number","doc_number","Query","doc_text","label"]
    writer.writerow(header)
    for row in dev_data_dict:
      # write the data
      writer.writerow(row)

