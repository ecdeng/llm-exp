import pandas as pd 

df = pd.read_csv("beer_reviews.csv", encoding='utf8', sep=',', on_bad_lines='skip', index_col=False, dtype='unicode')
df_2 = df.truncate(after = 1000)
# shortened = [example[:12] for example in df['text'].to_list()]
# df["truncated"] = shortened
df_2.to_csv("beer_reviews_short.csv", index=False, encoding="utf8")