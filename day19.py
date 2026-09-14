import pandas as pd

data = {
    "Applicant_ID": [101, 102, 103, 104],
    "Employment_Type": ["Salaried", "self-employed", "Salaried", "unemployed"],
}

df = pd.DataFrame(data)

print("--- Original data---")
print(df)

encoded_df = pd.get_dummies(df, columns=["Employment_Type"], dtype=int)


print(encoded_df)
