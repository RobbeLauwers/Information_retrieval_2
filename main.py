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
                if temp < 100:
                    temp += 1
                    dev_data_dict.append(row)


#Define the model. Either from scratch of by loading a pre-trained model (pretrained here)
# Note that this is a SentenceTransformer, not a crossencoder (which is what we eventually want)
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

#Define your train examples. (Trains on first 100 rows of dataset)
train_examples = []
for row in dev_data_dict:
  train_examples.append( InputExample( texts=[row[2],row[3]],label=float(row[4]) ) )

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100,save_best_model=True,output_path="./output/modelTest")

# Crossencoders are used to predict how well a query matches the result, which is what we need
model = CrossEncoder('./output/modelTest')

# Does this need more training?
model.fit(train_dataloader=train_dataloader, epochs=1, warmup_steps=100,save_best_model=True,output_path="./output/modelTest2")
scores = model.predict([["This is correct", "This is correct"],
                        ["This is not", "Gibberish"]])
print(scores)