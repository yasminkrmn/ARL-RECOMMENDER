############################################
# Project: Association Rule Learning Recommender
############################################



############################################
# Required Library and Functions
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from helper.helper import check_df, retail_data_prep

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


############################################
# 1: Data Preprocessing
############################################

# Online retail II data set

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
check_df(df)
df = retail_data_prep(df)
check_df(df)

############################################
# 2: Produce Association Rules for Germany Customers
############################################

rules_grm = create_rules(df, country="Germany")
# antecedent support: Tek başına X olasılığı
# consequent support: Tek başına Y olasılığı
# support: İkisinin birlikte görülme olasılığı
# confidence: X alındığında Y alınma olasılığı.
# lift: X alındığında Y alınma olasılığı .. kat artar.
# conviction: Y olmadan X'in beklenen frekansı

############################################
# 3: What are the names of the products whose IDs are given?
############################################

check_id(df, 21987)
# (PACK OF 6 SKULL PAPER CUPS)
check_id(df, 23235)
# (STORAGE TIN VINTAGE LEAF)
check_id(df, 22747)
# (POPPY'S PLAYHOUSE BATHROOM)


############################################
# 4: Make a Product Recommendation for Users in the Cart
############################################

# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747

arl_recommender(rules_grm, 21987, 1)
arl_recommender(rules_grm, 23235, 1)
arl_recommender(rules_grm, 22747, 1)


############################################
# 5: What are the Names of the Recommended Products?
############################################

check_id(df, arl_recommender(rules_grm, 21987, 1)[0])
check_id(df, arl_recommender(rules_grm, 23235, 1)[0])
check_id(df, arl_recommender(rules_grm, 22747, 1)[0])
