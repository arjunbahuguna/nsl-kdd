from collections import defaultdict
import os

#STEP 1: LOAD DATA
dataset_root = 'nsl-kdd'
header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']

category = defaultdict(list)
category['benign'].append('normal')

with open('training_attack_types.txt', 'r') as f:
	for line in f.readlines():
		attack, cat = line.strip().split(' ')
		category[cat].append(attack)
#print(category)

attack_mapping = dict((v,k) for k in category for v in category[k])
#print(attack_mapping)

#load train/test files
train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')


#READ TRAINING DATA
import pandas as pd
train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'] \
	.map(lambda x: attack_mapping[x])
train_df.drop(['success_pred'], axis=1, inplace=True)

test_df = pd.read_csv(train_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'] \
.map(lambda x: attack_mapping[x])
test_df.drop(['success_pred'], axis=1, inplace=True)




###STEP 2: UNDERSTAND DATA (visualize)
import matplotlib.pyplot as plt
train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()
train_attack_types.plot(kind='barh')
plt.show()
train_attack_cats.plot(kind='barh')
plt.show()




#DATA PREPARATION
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category','attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category','attack_type'], axis=1)

feature_names = defaultdict(list)

with open('kddcup.names.txt', 'r') as f:
	for line in f.readlines()[1:]:
		name, nature = line.strip()[:-1].split(': ')
		feature_names[nature].append(name)

print(feature_names)




#FEATURE ENGINEERING
# Concatenate DataFrames
combined_df_raw = pd.concat([train_x_raw, test_x_raw])
# Keep track of continuous, binary, and nominal features
continuous_features = feature_names['continuous']
continuous_features.remove('root_shell')
binary_features = ['land','logged_in','root_shell', 'su_attempted','is_host_login', 'is_guest_login']

nominal_features = list(set(feature_names['symbolic']) - set(binary_features))

# Generate dummy variables
combined_df = pd.get_dummies(combined_df_raw, \
columns=feature_names['symbolic'], \
drop_first=True)

# Separate into training and test sets again
train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

# Keep track of dummy variables
dummy_variables = list(set(train_x)-set(combined_df_raw))

#train_x.describe()



#NORMALIZATION (src_bytes > num_failed_logs by 10^7)

#what is normalization? show pictures

from sklearn.preprocessing import StandardScaler
# Fit StandardScaler to the training data
standard_scaler = StandardScaler().fit(train_x[continuous_features])
# Standardize training data
train_x[continuous_features] = \
	standard_scaler.transform(train_x[continuous_features])
# Standardize test data with scaler fitted to training data
test_x[continuous_features] = \
	standard_scaler.transform(test_x[continuous_features])





#STEP 5: CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(train_x, train_Y)
pred_y = classifier.predict(test_x)
results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)

#STEP 6: Check results
print(results)
print(error)