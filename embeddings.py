import sentence_transformers.cross_encoder.evaluation
from sentence_transformers import SentenceTransformer, InputExample, util, losses, CrossEncoder, cross_encoder, evaluation
from torch.utils.data import DataLoader
import csv
import math
import torch

# TODO: verwijderen als we volledige dataset inlezen
temp = 1

#Read data, list contains more lists: [query,document_text,relevant]
dev_data_dict = []

dev_data_0 = {}
dev_data_1 = {}

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
                if temp < 20:
                    temp += 1
                    dev_data_dict.append(row)

#Define your train examples. (Trains on first temp rows of dataset)
train_examples = []
sentence_pairs = []
labels = []
corpus = {}
for row in dev_data_dict:


    # Make corpus containing all queries
    if row[2] not in corpus:
        corpus[row[2]] = []

        # Make dictionaries containing {key: queryNumber, value: [documentNumber]} but separate dicts for labels 0 and 1
        # Used later for recall/precision
    if int(row[0]) not in dev_data_1:
        dev_data_1[int(row[0])] = []
    if int(row[0]) not in dev_data_0:
        dev_data_0[int(row[0])] = []
    if int(row[4]) == 1:
        dev_data_1[int(row[0])].append(int(row[1]))
    else:
        dev_data_0[int(row[0])].append(int(row[1]))
    #TODO: in the data where results are rated relevant or not, their order is also from most to least relevant.
    # We should take this into account by giving a small penalty to label based on position
    train_examples.append( InputExample( texts=[str(row[2]),str(row[3])],label=float(row[4]) ) )
    sentence_pairs.append([str(row[2]),str(row[3])])
    labels.append(float(row[4]))

    #If document not in corpus, add it
    if row[3] not in corpus[row[2]]:
        corpus[row[2]].append(row[3])

# Asymmetric search: ms marco models https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search
model = SentenceTransformer('msmarco-MiniLM-L-6-v3')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Does this need more training?
#model.fit(train_objectives=[(train_dataloader,train_loss)], epochs=1, warmup_steps=100, save_best_model=True, output_path="output/no_cross")

# TODO: Read data to make predictions about (list of lists each containing [query,document_text]

# TODO: actually use the correct data here
# Make predictions (scores is a list of integers

scores = []
for row in train_examples:
    query = row.texts[0]
    query_embedding = model.encode(query, convert_to_tensor=True)

    #TODO: is this a correct approach?
    # Take only the corpus that was actually retrieved for this query
    corpus_temp = corpus[query]
    corpus_embeddings = model.encode(corpus_temp, convert_to_tensor=True)
    top_k = len(corpus_temp)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus_temp[idx.item()], "(Score: " + str(float(score)) + ")")
        scores.append(float(score))


output_binary_label_unsorted = [[dev_data_dict[i][0], dev_data_dict[i][1], math.floor(scores[i][0] + 0.55)] for i in range(len(dev_data_dict))]

print(scores)

#(sorted([[dev_data_dict[i][0], dev_data_dict[i][1], math.floor(scores[i] + 0.5)] for i in range(len(dev_data_dict))], key=lambda x: (x[0], -x[2], x[1])))
#scores_query = [[dev_data_dict[i][0], dev_data_dict[i][1], math.floor(scores[i] + 0.5)] for i in range(len(dev_data_dict))]

# TODO: use same output format as dev_data.csv, label should probably be 1 or 0 instead of score

# TODO: actually sort results per query based on scores

#scores_sorted = sorted(scores_query, key=lambda x: (x[0], -x[2], x[1]))

#expected_scores = [[row[0], row[1], int(row[4])] for row in dev_data_dict]
#expected_scores = sorted(expected_scores, key=lambda x: (x[0], -x[2], x[1]))

# Based on code from https://towardsdatascience.com/rbo-v-s-kendall-tau-to-compare-ranked-lists-of-items-8776c5182899
def rbo(in1, in2, p=0.9):
    # tail recursive helper function
    list1 = [row[1] for row in in1]
    list2 = [row[1] for row in in2]
    def helper(ret, j, start, d):
        result = ret
        for i in range(j, d + 1):
            l1 = set(list1[start:start+i]) if start + i < len(list1) else set(list1[start:])
            l2 = set(list2[start:start+i]) if start + i < len(list2) else set(list2[start:])
            a_d = len(l1.intersection(l2))/i
            term = math.pow(p, i) * a_d
            result = ret + term
        return result
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

def precision_recall(dict_0,dict_1,results):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # precision = (aantal resultaten met label 1 die in testdata ook label 1 hebben)/(aantal resultaten met label 1)
    # recall = (aantal resultaten met label 1 die in testdata ook label 1 hebben)/(aantal in testdata met label 1)
    for row in results:
        # positive
        # If label is 1 (positive)
        if int(row[2]) == 1:
            # If document in this positive result is indeed in positive results for this query
            if int(row[1]) in dict_1[int(row[0])]:
                TP += 1
            else:
                FP += 1
        # negative
        else:
            # If document in this negative result is indeed in negative results
            if int(row[1]) in dict_0[int(row[0])]:
                TN += 1
            else:
                FN += 1
    # precision , recall
    # https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall
    print("TP: " + str(TP) + ", TN: " + str(TN) + ", FP: " + str(FP) + ", FN: " + str(FN) )
    return [( (TP)/(TP+FP) ),( (TP)/(TP+FN) )]


#print(expected_scores)
#print(scores_sorted)
#print(rbo(expected_scores, scores_sorted))
print("precision, recall: " + str(precision_recall(dev_data_0,dev_data_1,output_binary_label_unsorted)))