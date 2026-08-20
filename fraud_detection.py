import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
%matplotlib inline

# Load sample .pkl files
data_dir = 'data'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
dfs = [pd.read_pickle(os.path.join(data_dir,f)) for f in files]
data = pd.concat(dfs, ignore_index=True)
print('Loaded rows:', len(data))
data.head()

# Basic feature engineering
df = data.copy()
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['flag_large_amount'] = (df['TX_AMOUNT'] > 220).astype(int)
cust_agg = df.groupby('CUSTOMER_ID').TX_AMOUNT.agg(['count','mean','max']).reset_index().rename(columns={'count':'cust_tx_count','mean':'cust_avg_amount','max':'cust_max_amount'})
df = df.merge(cust_agg, on='CUSTOMER_ID', how='left')
term_agg = df.groupby('TERMINAL_ID').TX_AMOUNT.agg(['count','mean','max']).reset_index().rename(columns={'count':'term_tx_count','mean':'term_avg_amount','max':'term_max_amount'})
df = df.merge(term_agg, on='TERMINAL_ID', how='left')
df['hour'] = df['TX_DATETIME'].dt.hour
features = ['TX_AMOUNT','flag_large_amount','cust_tx_count','cust_avg_amount','cust_max_amount','term_tx_count','term_avg_amount','term_max_amount','hour']
df = df.dropna(subset=['TX_FRAUD'])
X = df[features].fillna(0)
y = df['TX_FRAUD'].astype(int)
print('Features shape:', X.shape, 'Fraud rate:', y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob))

# Feature importance
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print(feat_imp)
feat_imp.plot(kind='bar', figsize=(8,4))
plt.title('Feature Importances')
plt.show()
