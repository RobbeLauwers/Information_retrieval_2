from sentence_transformers import SentenceTransformer, InputExample, losses, CrossEncoder
from torch.utils.data import DataLoader
import csv
import math

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
scores = model.predict([[row[2], row[2]] for row in dev_data_dict])
scores_query = [[dev_data_dict[i][0], math.floor(scores[i] + 0.5)] for i in range(len(dev_data_dict))]

# TODO: use same output format as dev_data.csv, label should probably be 1 or 0 instead of score

# TODO: actually sort results per query based on scores

scores_sorted = sorted(scores_query, key=lambda x: (x[0], -x[1]))

expected_scores = [[row[0], int(row[4])] for row in dev_data_dict]

# Based on code from https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
def rbo(in1, in2, p=0.9):
    # tail recursive helper function
    list1 = [row[1] for row in in1]
    list2 = [row[1] for row in in2]
    def helper(ret, i, start, d):
        l1 = set(list1[start:start+i]) if start + i < len(list1) else set(list1[start:])
        l2 = set(list2[start:start+i]) if start + i < len(list2) else set(list2[start:])
        a_d = len(l1.intersection(l2))/i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, start, d)
    result_list = []
    start = 0
    k = 0
    current_query = in1[0][0]
    for i in range(len(list1)):
        if current_query != in1[i][0] or i == len(list1) - 1:
            x_k = len(set(list1[start:start+k]).intersection(set(list2[start:start+k])))
            summation = helper(0, 1, start, k)
            result_list.append(((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation))
            start = k
            current_query = in1[i][0]
        k += 1
    return sum(result_list) / len(result_list)

print(rbo(expected_scores, scores_sorted))