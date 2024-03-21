from itertools import combinations

def generate_itemsets(data, min_support):
    itemsets = {}
    support_count = {}
    total_transactions = len(data)

    for transaction in data:
        for item in transaction:
            if item in support_count:
                support_count[item] += 1
            else:
                support_count[item] = 1

    for item, count in support_count.items():
        support = count / total_transactions
        if support >= min_support:
            itemsets[frozenset([item])] = support

    return itemsets

def generate_candidate_itemsets(prev_itemsets, k):
    candidates = set()
    for itemset1 in prev_itemsets:
        for itemset2 in prev_itemsets:
            if len(itemset1.union(itemset2)) == k:
                candidates.add(itemset1.union(itemset2))
    return candidates

def prune_itemsets(candidate_itemsets, prev_itemsets, k):
    pruned_itemsets = set()
    for itemset in candidate_itemsets:
        subsets = combinations(itemset, k-1)
        if all(frozenset(subset) in prev_itemsets for subset in subsets):
            pruned_itemsets.add(itemset)
    return pruned_itemsets

def apriori(data, min_support):
    itemsets = {}
    k = 1
    while True:
        if k == 1:
            itemsets[k] = generate_itemsets(data, min_support)
        else:
            candidate_itemsets = generate_candidate_itemsets(itemsets[k-1], k)
            pruned_itemsets = prune_itemsets(candidate_itemsets, itemsets[k-1], k)
            if not pruned_itemsets:
                break
            itemsets[k] = {}
            for itemset in pruned_itemsets:
                count = sum(1 for transaction in data if itemset.issubset(transaction))
                support = count / len(data)
                if support >= min_support:
                    itemsets[k][itemset] = support
            if not itemsets[k]:
                break
        k += 1
    return itemsets

def generate_rules(itemsets, min_confidence):
    rules = []
    for k, itemsets_k in itemsets.items():
        if k < 2:
            continue
        for itemset, support in itemsets_k.items():
            for i in range(1, k):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = itemsets[len(antecedent)][antecedent]
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

# Sample transactions
transactions = [
    {'A', 'B', 'C'},
    {'A', 'C'},
    {'A', 'D'},
    {'B', 'E', 'F'}
]

min_support = 0.5
min_confidence = 0.5

itemsets = apriori(transactions, min_support)
rules = generate_rules(itemsets, min_confidence)

print("Frequent itemsets:")
for k, itemsets_k in itemsets.items():
    print(f"Itemsets of size {k}:")
    for itemset, support in itemsets_k.items():
        print(f"{itemset}: Support = {support}")

print("\nAssociation Rules:")
for rule in rules:
    antecedent, consequent, confidence = rule
    print(f"{antecedent} => {consequent}: Confidence = {confidence}")