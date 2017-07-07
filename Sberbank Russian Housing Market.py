#-----------Model_0------------#

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
id_test = test.id

#clean columns whose important feature is NAN in train_data while the opposite in test_data
indexs = []
for index, row in train.iterrows():
    if (math.isnan(row['max_floor']) or math.isnan(row['material']) or math.isnan(row['num_room']) or math.isnan(row['kitch_sq'])
        or math.isnan(row['state']) or math.isnan(row['full_sq'])
        ):
        indexs.append(index)
train = train.drop(train.index[indexs])

for f in train.columns:
    if train[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))

for f in test.columns:
    if test[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test[f].values)) 
        test[f] = lbl.transform(list(test[f].values))

test_features_to_remove = [ "id","timestamp",
                            'mosque_count_3000', 'oil_chemistry_raion', 'mosque_count_2000', '0_13_male', 'mosque_count_1000', '0_6_all',
                            'incineration_raion', 'mosque_count_1500', 'mosque_count_500', 'big_market_raion', '0_13_all', '0_17_all', 
                            'young_male','0_17_male', '0_6_female', '0_13_female', 'water_1line', 'market_count_500', 'nuclear_reactor_raion',
                            '7_14_all', 'radiation_raion', '0_17_female', 'thermal_power_plant_raion', 'railroad_1line', '7_14_female',
                            'market_count_1000', '0_6_male', '7_14_male', 'mosque_count_5000', 'cafe_count_2000_price_high', 'young_female'
                          ]
train_features_to_remove = ['id', 'price_doc', 'timestamp',
                            'mosque_count_3000', 'oil_chemistry_raion', 'mosque_count_2000', '0_13_male', 'mosque_count_1000', '0_6_all',
                            'incineration_raion', 'mosque_count_1500', 'mosque_count_500', 'big_market_raion', '0_13_all', '0_17_all', 
                            'young_male','0_17_male', '0_6_female', '0_13_female', 'water_1line', 'market_count_500', 'nuclear_reactor_raion',
                            '7_14_all', 'radiation_raion', '0_17_female', 'thermal_power_plant_raion', 'railroad_1line', '7_14_female',
                            'market_count_1000', '0_6_male', '7_14_male', 'mosque_count_5000', 'cafe_count_2000_price_high', 'young_female'
                           ]


train_y_all = train.price_doc.values
train_x_all = train.drop(train_features_to_remove, axis = 1)
test_id = test['id']
test_x = test.drop(test_features_to_remove, axis = 1)
dtrain_all = xgb.DMatrix(train_x_all, train_y_all)
dtest = xgb.DMatrix(test_x)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

#cross-validation1
# cv_output = xgb.cv(xgb_params, dtrain_all, num_boost_round = 1000, nfold = 3, early_stopping_rounds = 20, verbose_eval = 10, show_stdv = False)
# cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
# cv_rounds = len(cv_output)

model_0 = xgb.train(dict(xgb_params, silent=0), dtrain_all, num_boost_round = 262)
test_y = model_0.predict(dtest)
output_0 = pd.DataFrame({'id': test_id, 'price_doc': test_y})



#-----------Model_1------------#
train_df = pd.read_csv("./input/train.csv", parse_dates=['timestamp'])
test_df = pd.read_csv("./input/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("./input/macro.csv", parse_dates=['timestamp'])

mult = 0.969
y_train = train_df['price_doc'].values * mult + 10
id_test = test_df['id']

train_df.drop(['id', 'price_doc'], axis=1, inplace=True)
test_df.drop(['id'], axis=1, inplace=True)

num_train = len(train_df)
df_all = pd.concat([train_df, test_df])

df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 365)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 365)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dayofweek'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

df_all['building_age'] = 2020 - df_all['build_year']

df_all['value'] = df_all['building_age'] * df_all['full_sq']


train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]
test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])
   train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])
   train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')

def add_time_features(col):
   col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])
   test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())

   col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])
   test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())

add_time_features('building_name')
add_time_features('sub_area')


df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)


factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)

num_boost_rounds = 466
model_1 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model_1.predict(dtest)

output_1 = pd.DataFrame({'id': id_test, 'price_doc': y_pred})


# In[ ]:


#-----------Model_2------------#
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
train_lat_lon = pd.read_csv('../input/train_lat_lon.csv')
test_lat_lon = pd.read_csv('../input/test_lat_lon.csv')
train = pd.merge(train, train_lat_lon, how='left', on='id')
test = pd.merge(test, test_lat_lon, how='left', on='id')
id_test = test.id

bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN

equal_index = test[test.life_sq == 10 * test.full_sq].index
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]

bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq > 5000].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN

bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.full_sq > 5000].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN

bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[train.kitch_sq >= 1000].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= 1000].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN


bad_index = train[(train.full_sq > 220) & (train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.loc[bad_index, "full_sq"] = np.NaN

bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN

train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)

for i, row in train.iterrows():
    if row['build_year'] == 0.0 or row['build_year'] == 1.0 or row['build_year'] == 3.0:
        train.loc[i, 'build_year'] = np.NaN
    if row['build_year'] == 20052009.0:
        train.loc[i, 'build_year'] = 2007.0
    if row['build_year'] == 215.0:
        train.loc[i, 'build_year'] = 2015.0
    if row['build_year'] == 20.0:
        train.loc[i, 'build_year'] = 2000.0
    if row['build_year'] == 71.0:
        train.loc[i, 'build_year'] = 1971.0
    if row['build_year'] == 4965.0:
        train.loc[i, 'build_year'] = 1965.0
    if row['kitch_sq'] == 1970.0:
        train.loc[i, 'build_year'] = 1970.0
        
for i, row in test.iterrows():
    if row['build_year'] == 0.0 or row['build_year'] == 1.0 or row['build_year'] == 3.0:
        test.loc[i, 'build_year'] = np.NaN
    if row['build_year'] == 20052009.0:
        test.loc[i, 'build_year'] = 2007.0
    if row['build_year'] == 215.0:
        test.loc[i, 'build_year'] = 2015.0
    if row['build_year'] == 20.0:
        test.loc[i, 'build_year'] = 2000.0
    if row['build_year'] == 71.0:
        test.loc[i, 'build_year'] = 1971.0
    if row['build_year'] == 4965.0:
        test.loc[i, 'build_year'] = 1965.0

bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN

bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN

bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = 3
train['state'] = train['state'].fillna(train['state'].mode().loc[0])
test['state'] = test['state'].fillna(test['state'].mode().loc[0])

train['material'] = train['material'].fillna(train['material'].mode().loc[0])
test['material'] = test['material'].fillna(test['material'].mode().loc[0])

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 365)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 365)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 365)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 365)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dayofweek'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dayofweek'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)

train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = (train["life_sq"] - train["kitch_sq"]) / train['num_room'].astype(float)
test['room_size'] = (test["life_sq"] - test["kitch_sq"]) / test['num_room'].astype(float)
train.loc[train['room_size'] < 0, 'room_size'] = np.nan
test.loc[test['room_size'] < 0, 'room_size'] = np.nan

#building_age = 2020 - Build_Year
train["building_age"] = 2020.0 - train["build_year"]
test["building_age"] = 2020.0 - test["build_year"]
train.loc[train['building_age'] < 0, 'building_age'] = np.nan
test.loc[test['building_age'] < 0, 'building_age'] = np.nan

#Non-residential area to living area ratio
train["non_residential_ratio"] = (train["full_sq"] - train["life_sq"])/train["full_sq"]
test["non_residential_ratio"] = (test["full_sq"] - test["life_sq"])/test["full_sq"]

#value of building by age and area
train['value'] = train['building_age'] * train['full_sq']
test['value'] = test['building_age'] * test['full_sq']

rate_2015_q2 = 1
rate_2015_q1 = rate_2015_q2 / 0.9932
rate_2014_q4 = rate_2015_q1 / 1.0112
rate_2014_q3 = rate_2014_q4 / 1.0169
rate_2014_q2 = rate_2014_q3 / 1.0086
rate_2014_q1 = rate_2014_q2 / 1.0126
rate_2013_q4 = rate_2014_q1 / 0.9902
rate_2013_q3 = rate_2013_q4 / 1.0041
rate_2013_q2 = rate_2013_q3 / 1.0044
rate_2013_q1 = rate_2013_q2 / 1.0104  # This is 1.002 (relative to mult), close to 1:
rate_2012_q4 = rate_2013_q1 / 0.9832  # maybe use 2013q1 as a base quarter and get rid of mult?
rate_2012_q3 = rate_2012_q4 / 1.0277
rate_2012_q2 = rate_2012_q3 / 1.0279
rate_2012_q1 = rate_2012_q2 / 1.0279
rate_2011_q4 = rate_2012_q1 / 1.076
rate_2011_q3 = rate_2011_q4 / 1.0236
rate_2011_q2 = rate_2011_q3 / 1
rate_2011_q1 = rate_2011_q2 / 1.011


# train 2015
train['average_q_price'] = 1

train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2

train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1


# train 2014
train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4

train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3

train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2

train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1


# train 2013
train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4

train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3

train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2

train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1


# train 2012
train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4

train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3

train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2

train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1


# train 2011
train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index
train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4

train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index
train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3

train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index
train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2

train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index
train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1

train['price_doc'] = train['price_doc'] * train['average_q_price']


# In[8]:


mult = 1.055 # Trying another magic number
train['price_doc'] = train['price_doc'] * mult
y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]


# In[10]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 430
model_2 = xgb.train(dict(xgb_params, silent = 0), dtrain, num_boost_round = num_boost_rounds)

y_predict = model_2.predict(dtest)
output_2 = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


# In[ ]:


#-----------Model_3------------#
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
id_test = test_df.id


y_train = train_df["price_doc"]
x_train = train_df.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test_df.drop(["id", "timestamp"], axis=1)

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

num_boost_rounds = 415
model_3 = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model_3.predict(dtest)
output_3 = pd.DataFrame({'id': id_test, 'price_doc': y_predict})


# In[ ]:


#-----------Model Averaging------------#
result_1 = output_1.merge(output_0, on="id", suffixes=['_1','_0'])
result_1["price_doc"] = np.exp( .75*np.log(result_1.price_doc_1) +
                                .25*np.log(result_1.price_doc_0) )
result_2 = result_1.merge(output_2, on="id", suffixes=['_0_1','_2'])
result_2["price_doc"] = np.exp( .80*np.log(result_2.price_doc_0_1) +
                                .20*np.log(result_2.price_doc_2) )
result_3 = result_2.merge(output_3, on="id", suffixes=['_0_1_2','_3'])
result_3["price_doc"] = np.exp( .73*np.log(result_3.price_doc_0_1_2) +
                                .27*np.log(result_3.price_doc_3) )

result_3["price_doc"] = result_3["price_doc"] *0.9915        
result_3.drop(["price_doc_0","price_doc_1","price_doc_2","price_doc_3","price_doc_0_1","price_doc_0_1_2", ],axis=1,inplace=True)
result_3.to_csv('final-result.csv', index=False)

