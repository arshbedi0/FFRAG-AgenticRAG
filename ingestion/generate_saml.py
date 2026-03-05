import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── CONFIG ──
N_TOTAL = 1000
N_SUSPICIOUS = 280  # ~28% suspicious, realistic for AML datasets

# ── REFERENCE DATA ──
BANK_LOCATIONS = {
    "UK": 0.40,
    "UAE": 0.15,
    "Turkey": 0.10,
    "Mexico": 0.08,
    "Morocco": 0.07,
    "India": 0.06,
    "Pakistan": 0.05,
    "Nigeria": 0.04,
    "Germany": 0.03,
    "USA": 0.02,
}

CURRENCY_MAP = {
    "UK": "UK pounds",
    "UAE": "Dirham",
    "Turkey": "Turkish lira",
    "Mexico": "Mexican peso",
    "Morocco": "Moroccan dirham",
    "India": "Indian rupee",
    "Pakistan": "Pakistani rupee",
    "Nigeria": "Nigerian naira",
    "Germany": "Euro",
    "USA": "US dollar",
}

PAYMENT_TYPES = ["Credit card", "Debit card", "Cash", "ACH transfer", "Cross-border", "Cheque"]

# Suspicious typologies matching SAML-D
TYPOLOGIES = [
    "Structuring",           # Breaking transactions just under reporting threshold
    "Layering",              # Rapid movement through multiple accounts
    "Currency_Mismatch",     # Payment/received currency mismatch with high-risk country
    "Smurfing",              # Multiple small deposits to aggregate
    "Round_Trip",            # Money leaving and returning
    "Dormant_Reactivation",  # Long-dormant account suddenly active
    "High_Risk_Corridor",    # UK → UAE / UK → Turkey corridor
    "Rapid_Succession",      # Many transactions in short time window
]

# ── ACCOUNT POOL ──
# Create a realistic pool — some accounts appear many times (hubs)
def gen_account_pool(n=300):
    accounts = [str(random.randint(100_000_000, 999_999_999)) for _ in range(n)]
    return accounts

ACCOUNT_POOL = gen_account_pool(300)

# Designate some accounts as "hub" accounts that appear frequently (suspicious pattern)
HUB_ACCOUNTS = random.sample(ACCOUNT_POOL, 15)

# ── HELPER FUNCTIONS ──
def weighted_location():
    locs = list(BANK_LOCATIONS.keys())
    weights = list(BANK_LOCATIONS.values())
    return random.choices(locs, weights=weights, k=1)[0]

def currency_for(location, mismatch=False):
    if mismatch:
        # Return a currency that doesn't match location
        wrong = [c for l, c in CURRENCY_MAP.items() if l != location]
        return random.choice(wrong)
    return CURRENCY_MAP[location]

def gen_time(base_date, seconds_offset):
    t = datetime(2022, 10, 7, 10, 35, 0) + timedelta(seconds=seconds_offset)
    return t.strftime("%H:%M:%S"), t.strftime("%Y-%m-%d")

# ── GENERATE NORMAL TRANSACTIONS ──
def normal_transaction(idx):
    sender_loc = weighted_location()
    receiver_loc = weighted_location()
    sender_acc = random.choice(ACCOUNT_POOL)
    receiver_acc = random.choice([a for a in ACCOUNT_POOL if a != sender_acc])
    amount = round(np.random.lognormal(7.5, 1.2), 2)  # realistic spread, mostly £500-£20k
    amount = min(amount, 85000)
    payment_type = random.choices(
        PAYMENT_TYPES,
        weights=[0.25, 0.30, 0.10, 0.20, 0.10, 0.05]
    )[0]
    time_str, date_str = gen_time(None, idx * random.randint(2, 8))
    return {
        "Time": time_str,
        "Date": date_str,
        "Sender_account": sender_acc,
        "Receiver_account": receiver_acc,
        "Amount": amount,
        "Payment_currency": currency_for(sender_loc),
        "Received_currency": currency_for(receiver_loc),
        "Sender_bank_location": sender_loc,
        "Receiver_bank_location": receiver_loc,
        "Payment_type": payment_type,
        "Is_suspicious": 0,
        "Type": "Normal",
    }

# ── GENERATE SUSPICIOUS TRANSACTIONS (with typology logic) ──
def suspicious_transaction(idx, typology=None):
    if typology is None:
        typology = random.choice(TYPOLOGIES)

    sender_loc = weighted_location()
    receiver_loc = weighted_location()
    sender_acc = random.choice(HUB_ACCOUNTS)  # hub account involved
    receiver_acc = random.choice(ACCOUNT_POOL)
    payment_type = random.choice(PAYMENT_TYPES)
    time_str, date_str = gen_time(None, idx * random.randint(1, 4))

    amount = round(np.random.lognormal(7.5, 1.2), 2)

    # ── Typology-specific logic ──
    if typology == "Structuring":
        # Just under $10,000 reporting threshold
        amount = round(random.uniform(8500, 9999), 2)
        payment_type = random.choice(["Cash", "Cheque"])

    elif typology == "Smurfing":
        # Multiple small amounts that aggregate
        amount = round(random.uniform(500, 2500), 2)
        sender_acc = random.choice(HUB_ACCOUNTS)

    elif typology == "Layering":
        # Large amount moving rapidly
        amount = round(random.uniform(20000, 75000), 2)
        payment_type = "ACH transfer"

    elif typology == "Currency_Mismatch":
        # Currency doesn't match the bank location
        receiver_loc = random.choice(["UAE", "Turkey", "Morocco"])
        # Mismatch: bank in UAE but received in UK pounds
        pass  # handled below

    elif typology == "High_Risk_Corridor":
        sender_loc = "UK"
        receiver_loc = random.choice(["UAE", "Turkey", "Morocco", "Nigeria"])
        amount = round(random.uniform(5000, 50000), 2)
        payment_type = "Cross-border"

    elif typology == "Dormant_Reactivation":
        # Very large amount from an otherwise inactive account
        amount = round(random.uniform(30000, 100000), 2)
        sender_acc = random.choice(ACCOUNT_POOL)  # non-hub, fresh account

    elif typology == "Round_Trip":
        # Sender and receiver are in same location, large amount
        receiver_loc = sender_loc
        amount = round(random.uniform(15000, 60000), 2)

    elif typology == "Rapid_Succession":
        # Small-medium amount but will appear in clusters
        amount = round(random.uniform(1000, 8000), 2)

    # Currency mismatch logic for all suspicious types
    currency_mismatch = typology == "Currency_Mismatch" or (random.random() < 0.3 and receiver_loc in ["UAE", "Turkey", "Morocco"])
    received_currency = currency_for(receiver_loc, mismatch=currency_mismatch)

    return {
        "Time": time_str,
        "Date": date_str,
        "Sender_account": sender_acc,
        "Receiver_account": receiver_acc,
        "Amount": amount,
        "Payment_currency": currency_for(sender_loc),
        "Received_currency": received_currency,
        "Sender_bank_location": sender_loc,
        "Receiver_bank_location": receiver_loc,
        "Payment_type": payment_type,
        "Is_suspicious": 1,
        "Type": typology,
    }

# ── BUILD DATASET ──
records = []

# Normal transactions
for i in range(N_TOTAL - N_SUSPICIOUS):
    records.append(normal_transaction(i))

# Suspicious — distribute across typologies
typology_counts = {t: N_SUSPICIOUS // len(TYPOLOGIES) for t in TYPOLOGIES}
# Fill remainder
remainder = N_SUSPICIOUS - sum(typology_counts.values())
for t in list(TYPOLOGIES)[:remainder]:
    typology_counts[t] += 1

for typology, count in typology_counts.items():
    for i in range(count):
        records.append(suspicious_transaction(len(records) + i, typology=typology))

# ── SHUFFLE & SORT BY TIME ──
random.shuffle(records)
df = pd.DataFrame(records)

# Re-sort by time to make it look like a real transaction log
df = df.sort_values("Time").reset_index(drop=True)

# ── STATS ──
print(f"Total transactions: {len(df)}")
print(f"Suspicious: {df['Is_suspicious'].sum()} ({df['Is_suspicious'].mean()*100:.1f}%)")
print(f"Normal: {(df['Is_suspicious']==0).sum()}")
print(f"\nTypology breakdown:")
print(df[df['Is_suspicious']==1]['Type'].value_counts().to_string())
print(f"\nSender bank location distribution:")
print(df['Sender_bank_location'].value_counts().to_string())
print(f"\nHigh-risk corridors (UK→UAE/Turkey/Morocco/Nigeria):")
hr = df[(df['Sender_bank_location']=='UK') & (df['Receiver_bank_location'].isin(['UAE','Turkey','Morocco','Nigeria']))]
print(f"  Count: {len(hr)}, Suspicious: {hr['Is_suspicious'].sum()}")
print(f"\nStructuring cases (amount £8500-£9999):")
struct = df[(df['Amount'] >= 8500) & (df['Amount'] <= 9999)]
print(f"  Count: {len(struct)}, Suspicious: {struct['Is_suspicious'].sum()}")
print(f"\nAmount stats:")
print(df['Amount'].describe().round(2).to_string())

# ── SAVE ──
df.to_csv("/mnt/user-data/outputs/saml_synthetic_1000.csv", index=False)
print("\nSaved to saml_synthetic_1000.csv")
