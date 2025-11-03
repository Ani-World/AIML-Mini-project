"""
Train Apriori association rules from movie metadata
Transactions are lists of [genre, actor, director] items
"""
import pandas as pd
import json
import os
import ast
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load processed dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} movies for Apriori training")

# Extract transactions (list of lists)
print("Extracting transactions...")
transactions = []

for idx, row in df.iterrows():
    # Parse apriori_items column
    try:
        items = ast.literal_eval(row['apriori_items'])
        # Filter out empty items
        items = [item.strip() for item in items if item and item.strip()]
        if items:
            transactions.append(items)
    except Exception as e:
        print(f"Error parsing row {idx}: {e}")
        continue

print(f"Extracted {len(transactions)} transactions")
print(f"Example transaction: {transactions[0][:5] if transactions else 'None'}")

# Encode transactions for Apriori
print("Encoding transactions...")
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Encoded {df_encoded.shape[0]} transactions with {df_encoded.shape[1]} unique items")

# Mine frequent itemsets
print("Mining frequent itemsets with Apriori...")
frequent_itemsets = apriori(
    df_encoded, 
    min_support=0.01,  # Itemsets must appear in at least 1% of transactions
    use_colnames=True,
    max_len=2  # Only pairs for simplicity
)

print(f"Found {len(frequent_itemsets)} frequent itemsets")

if len(frequent_itemsets) > 0:
    # Generate association rules
    print("Generating association rules...")
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.3  # Rules must have at least 30% confidence
    )
    
    print(f"Generated {len(rules)} association rules")
    
    # Filter and sort by confidence
    rules = rules[rules['confidence'] >= 0.3].sort_values('confidence', ascending=False)
    
    if len(rules) > 0:
        print("\nTop association rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence']].head(10))
    
    # Build recommendation mapping: {movie_id: [list of reasons]}
    movie_recommendations = {}
    
    # Create a mapping of items to movie_ids
    movie_id_to_items = {}
    for idx, row in df.iterrows():
        movie_id = idx + 1  # Assuming movie_id is index + 1
        try:
            items = ast.literal_eval(row['apriori_items'])
            items = [item.strip() for item in items if item and item.strip()]
            movie_id_to_items[movie_id] = set(items)
        except:
            continue
    
    # For each rule, find movies that match consequent
    for _, rule in rules.iterrows():
        consequent = rule['consequents']
        if len(consequent) == 1:
            consequent_item = list(consequent)[0]
            
            # Find all movies with this consequent item
            for movie_id, items in movie_id_to_items.items():
                if consequent_item in items:
                    antecedent = list(rule['antecedents'])
                    
                    # Build reason string
                    if len(antecedent) == 1:
                        reason = f"Users who like {antecedent[0]} also like {consequent_item}"
                    else:
                        reason = f"Users who like {', '.join(antecedent)} also like {consequent_item}"
                    
                    if movie_id not in movie_recommendations:
                        movie_recommendations[movie_id] = []
                    
                    # Only add if doesn't exist and confidence is decent
                    if reason not in movie_recommendations[movie_id] and rule['confidence'] >= 0.4:
                        movie_recommendations[movie_id].append(reason)
    
    # Save rules to JSON
    apriori_path = os.path.join(BASE_DIR, "models", "apriori_rules.json")
    with open(apriori_path, 'w') as f:
        json.dump(movie_recommendations, f, indent=2)
    
    print(f"\n[OK] Saved Apriori rules to {apriori_path}")
    print(f"Movies with recommendations: {len(movie_recommendations)}")
    
    # Show examples
    if movie_recommendations:
        sample_ids = list(movie_recommendations.keys())[:5]
        print("\nExample recommendations:")
        for movie_id in sample_ids:
            print(f"  Movie {movie_id}: {movie_recommendations[movie_id][:2]}")
else:
    print("[WARNING] No frequent itemsets found. Try lowering min_support or check your data.")

print("\n[OK] Apriori training complete!")

