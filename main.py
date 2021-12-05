from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
import csv

# TODO: verwijderen als we volledige dataset inlezen
temp = 1
#Read data
dev_data_dict = []
# https://docs.python.org/3/library/csv.html
csv.field_size_limit(991072)
with open('./data/dev_data.csv', newline='', encoding="utf8") as csvfile:
    dev_data_csv = csv.reader(csvfile, delimiter=',')
    for row in dev_data_csv:
        if row[0] != "Query_number":
            if len(row) != 5:
                print("error in input")
                print(row)
            else:
                #TODO: verwijderen als we volledige dataset inlezen
                if temp < 1000:
                    temp += 1
                    dev_data_dict.append(row)

#Define your train examples. (Trains on first 100 rows of dataset)
train_examples = []
for row in dev_data_dict:
  train_examples.append( InputExample( texts=[row[2],row[3]],label=float(row[4]) ) )

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Crossencoders are used to predict how well a query matches the result, which is what we need
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')

# Does this need more training?
model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100,save_best_model=True,output_path="./output/modelTest")
scores = model.predict([["This is correct", "This is correct"],
                        ["This is not", "Gibberish"]])
print(scores)