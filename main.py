from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
import csv

# TODO: verwijderen als we volledige dataset inlezen
temp = 1

#Read data, list contains more lists: [query,document_text,relevant]
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

#Define your train examples. (Trains on first temp rows of dataset)
train_examples = []
for row in dev_data_dict:
    #TODO: in the data where results are rated relevant or not, their order is also from most to least relevant.
    # We should take this into account by giving a small penalty to label based on position
  train_examples.append( InputExample( texts=[row[2],row[3]],label=float(row[4]) ) )

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Crossencoders are used to predict how well a query matches the result, which is what we need
model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2')

# Does this need more training?
model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100,save_best_model=True,output_path="./output/modelTest")

# TODO: Read data to make predictions about (list of lists each containing [query,document_text]

# TODO: actually use the correct data here
# Make predictions (scores is a list of integers
scores = model.predict([["This is correct", "This is correct"],
                        ["This is not", "Gibberish"]])

# TODO: use same output format as dev_data.csv, label should probably be 1 or 0 instead of score

# TODO: actually sort results per query based on scores

# TODO: rank biased overlap