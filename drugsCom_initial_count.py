import pandas
drugsCom = pandas.read_csv('drugsCom_raw (1)/drugsComTrain_raw.tsv',parse_dates = ['date'], sep='\t')

fda = pandas.read_csv('drugs_recalls_fda.tsv', parse_dates = [0], sep='\t')
drugsCom.columns

unique_drugs = drugsCom['drugName'].unique()
len(unique_drugs)
unique_drugs[1]

drugsCom.size

total_products_recalled = 0
for online_drug in unique_drugs:
    total_products_recalled += fda['Product Description'].str.contains(online_drug, regex=False).sum()

print(total_products_recalled)
# 76

# # seperate out unique drugs 
# total_products_recalled = 0
# for online_drug in unique_drugs:
#     new_df = pandas.DataFrame()
#     for drug in online_drug.split(' / ')
#         new_df[drug] = fda['Product Description'].str.contains(drug, regex=False)
#     new_df.all(axis='columns').sum()

# print(total_products_recalled)

# use only primary active ingredient 
total_products_recalled = 0
for online_drug in unique_drugs:
    online_drug = online_drug.split(' / ')[0]
    total_products_recalled += fda['Product Description'].str.contains(online_drug, regex=False).sum()

print(total_products_recalled)


unique_products_recalled = 0
for online_drug in unique_drugs:
    if any(fda['Product Description'].str.contains(online_drug, regex=False)):
        print(online_drug)
        break
        unique_products_recalled += 1

print(unique_products_recalled)
# 58

# Total Relevant Review Count
unique_drugs = drugsCom.groupby(['drugName']).count().reset_index()
review_count = 0
for index, row in unique_drugs.iterrows():
    recalled_drugs = fda['Product Description'].str.contains(row[0], regex=False)
    if any(recalled_drugs):
        review_count += row[1]

print(review_count)
# 7083


# Filter reviews for before recall
recalled_reviews = pandas.DataFrame()
unique_drugs = drugsCom.groupby(['drugName'])
review_count = 0
for drug, row in unique_drugs:
    recalled_drugs = fda['Product Description'].str.contains(drug.split(' / ')[0], regex=False)
    # drug is recalled
    if any(recalled_drugs):
        review_count += len(row['review'])
        recalled_reviews = recalled_reviews.append(row[row['date'] < fda[recalled_drugs]['Date'].iloc[0]])
        # pandas.concat(recalled_reviews['recalled_reviews'],row['review'])



print(review_count)
# 7083

# 7694 reviews if using only first active ingredient


