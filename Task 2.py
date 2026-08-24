# ============================================================
# DecodeLabs Internship - Artificial Intelligence
# Project 2: Data Classification Using AI
# ============================================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from pypdf import PdfReader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("DecodeLabs Project 2 - Data Classification Using AI")
print("=" * 70)


# ============================================================
# 2. DATASET PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_FILE = os.path.join(
    BASE_DIR,
    "Dataset for Data Analytics.pdf"
)

CONFUSION_MATRIX_FILE = os.path.join(
    BASE_DIR,
    "confusion_matrix.png"
)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_FILE):
    print()
    print(f"ERROR: Dataset was not found.")
    print(f"Expected location:")
    print(DATASET_FILE)
    print()
    print("Make sure 'Dataset for Data Analytics.pdf' is inside")
    print("the Task 2 folder.")
    raise SystemExit


# ============================================================
# 4. LOAD PDF DATASET
# ============================================================

print()
print("Loading dataset...")

reader = PdfReader(DATASET_FILE)

pdf_text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        pdf_text += page_text + "\n"


# ============================================================
# 5. EXTRACT RECORDS FROM PDF
# ============================================================

"""
The PDF contains order records in the following logical format:

OrderID
CustomerID
Product
Quantity
UnitPrice
ShippingAddress
PaymentMethod
OrderStatus
TrackingNumber
ItemsInCart
CouponCode
ReferralSource
TotalPrice

The date field in the PDF extraction appears as ########,
so it is intentionally ignored during extraction.
"""

products = r"(?:Laptop|Phone|Tablet|Monitor|Printer|Chair|Desk)"
payment_methods = r"(?:Online|Debit Card|Credit Card|Gift Card|Cash)"
statuses = r"(?:Cancelled|Delivered|Pending|Returned|Shipped)"
coupons = r"(?:SAVE10|FREESHIP|WINTER15)"
referrals = r"(?:Instagram|Referral|Email|Facebook|Google)"

pattern = re.compile(
    rf"(ORD\d+)\s*#+\s*"
    rf"(C\d+)\s+"
    rf"({products})\s+"
    rf"(\d+)\s+"
    rf"([\d.]+)\s+"
    rf"(\d+)\s+Main\s+St"
    rf"({payment_methods})"
    rf"\s*"
    rf"({statuses})\s+"
    rf"(TRK\d+)\s+"
    rf"(\d+)\s+"
    rf"({coupons})?"
    rf"\s*"
    rf"({referrals})\s+"
    rf"([\d.]+)"
)


records = []

for match in pattern.finditer(pdf_text):

    (
        order_id,
        customer_id,
        product,
        quantity,
        unit_price,
        address_number,
        payment_method,
        order_status,
        tracking_number,
        items_in_cart,
        coupon_code,
        referral_source,
        total_price
    ) = match.groups()

    records.append({
        "OrderID": order_id,
        "CustomerID": customer_id,
        "Product": product,
        "Quantity": int(quantity),
        "UnitPrice": float(unit_price),
        "ShippingAddress": f"{address_number} Main St",
        "PaymentMethod": payment_method,
        "OrderStatus": order_status,
        "TrackingNumber": tracking_number,
        "ItemsInCart": int(items_in_cart),
        "CouponCode": coupon_code if coupon_code else "No Coupon",
        "ReferralSource": referral_source,
        "TotalPrice": float(total_price)
    })


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    print()
    print("ERROR: No records could be extracted from the PDF.")
    print("Please check the dataset format.")
    raise SystemExit


print()
print("Dataset loaded successfully.")
print(f"Number of records : {len(df)}")
print(f"Number of columns : {len(df.columns)}")


# ============================================================
# 7. DATASET OVERVIEW
# ============================================================

print()
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)

print()
print("Columns:")
print(list(df.columns))

print()
print("First 5 records:")
print(df.head().to_string(index=False))


# ============================================================
# 8. DATA QUALITY CHECK
# ============================================================

print()
print("Missing Values:")

missing_values = df.isnull().sum()

print(missing_values.to_string())

print()
print("Duplicate Records:", df.duplicated().sum())


# Remove exact duplicate records if any
df = df.drop_duplicates().reset_index(drop=True)


# ============================================================
# 9. TARGET DISTRIBUTION
# ============================================================

print()
print("Order Status Distribution:")

status_distribution = df["OrderStatus"].value_counts()

print(status_distribution.to_string())


# ============================================================
# 10. FEATURES AND TARGET
# ============================================================

print()
print("=" * 70)
print("FEATURES AND TARGET")
print("=" * 70)

"""
Identifiers such as OrderID, CustomerID and TrackingNumber
are not used as predictive features.

ShippingAddress is also excluded because the address number
does not provide a meaningful classification signal.

TotalPrice is derived from Quantity × UnitPrice, so the
original Quantity and UnitPrice are retained instead.
"""

features = [
    "Product",
    "Quantity",
    "UnitPrice",
    "PaymentMethod",
    "ItemsInCart",
    "CouponCode",
    "ReferralSource"
]

target = "OrderStatus"

X = df[features]
y = df[target]

print()
print("Input Features:")
print(features)

print()
print("Target:")
print(target)


# ============================================================
# 11. NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "Quantity",
    "UnitPrice",
    "ItemsInCart"
]

categorical_features = [
    "Product",
    "PaymentMethod",
    "CouponCode",
    "ReferralSource"
]


# ============================================================
# 12. TRAIN-TEST SPLIT
# ============================================================

print()
print("=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True,
    stratify=y
)

print()
print(f"Total samples    : {len(X)}")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

print()
print("Training percentage : 80%")
print("Testing percentage  : 20%")
print("Data shuffled       : Yes")
print("Stratified split    : Yes")


# ============================================================
# 13. PREPROCESSING
# ============================================================

"""
KNN is distance-based.

Therefore:
- Numerical features are standardized using StandardScaler.
- Categorical features are converted to numerical form using
  OneHotEncoder.
"""

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 14. KNN MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

K_VALUE = 5

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=K_VALUE
            )
        )
    ]
)

print()
print("Algorithm : K-Nearest Neighbors (KNN)")
print(f"K value   : {K_VALUE}")
print("Scaling   : StandardScaler")
print("Encoding  : OneHotEncoder")


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print()
print("Training model...")

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 16. MAKE PREDICTIONS
# ============================================================

print()
print("Making predictions on test data...")

y_pred = model.predict(X_test)

print("Prediction completed.")


# ============================================================
# 17. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 70)
print("MODEL ACCURACY")
print("=" * 70)

print()
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")


# ============================================================
# 18. CLASSIFICATION METRICS
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print()
print("=" * 70)
print("CLASSIFICATION METRICS")
print("=" * 70)

print()
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# 19. DETAILED CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=sorted(y.unique())
)

class_names = sorted(y.unique())

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print("Rows    = Actual Class")
print("Columns = Predicted Class")
print()

cm_dataframe = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print(cm_dataframe.to_string())


# ============================================================
# 21. SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(9, 7))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

plt.title(
    "DecodeLabs Project 2 - KNN Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print()
print("Dataset                : Dataset for Data Analytics.pdf")
print(f"Records used           : {len(df)}")
print("Classification model   : K-Nearest Neighbors")
print(f"K value                : {K_VALUE}")
print("Train/Test split       : 80% / 20%")
print("Data shuffled          : Yes")
print("Feature scaling        : StandardScaler")
print("Categorical encoding   : OneHotEncoder")
print(f"Accuracy               : {accuracy * 100:.2f}%")
print(f"Weighted Precision     : {precision:.4f}")
print(f"Weighted Recall        : {recall:.4f}")
print(f"Weighted F1 Score      : {f1:.4f}")

print()
print("Confusion matrix saved to:")
print(CONFUSION_MATRIX_FILE)

print()
print("=" * 70)
print("PROJECT 2 COMPLETED")
print("=" * 70)
