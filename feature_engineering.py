print("Gelişmiş özellik mühendisliği başlıyor...")

def feature_engineering(df):
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df = df.sort_values(['user_session', 'event_time'])
    df['time_since_last_event'] = df.groupby('user_session')['event_time'].diff().dt.total_seconds()
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Data birleştir
combined_df = pd.concat([train_df, test_df], sort=False, ignore_index=True)

user_agg = combined_df.groupby('user_id').agg(
    user_total_sessions=('user_session', 'nunique'),
    user_total_buys=('event_type', lambda x: (x == 'BUY').sum()),
    user_avg_session_duration=('time_since_last_event', 'sum'),
    user_distinct_products=('product_id', 'nunique')
)

user_agg['user_avg_session_duration'] /= user_agg['user_total_sessions']
combined_df = pd.merge(combined_df, user_agg, on='user_id', how='left')

product_agg = combined_df.groupby('product_id').agg(
    prod_views=('event_type', lambda x: (x == 'VIEW').sum()),
    prod_buys=('event_type', lambda x: (x == 'BUY').sum())
).reset_index()

product_agg['prod_buy_ratio'] = product_agg['prod_buys'] / (product_agg['prod_views'] + 1e-6)
combined_df = pd.merge(combined_df, product_agg, on='product_id', how='left')


def aggregate_session_features(df):
    agg_funcs = {
        'event_type': ['count'],
        'product_id': ['nunique'],
        'category_id': ['nunique'],
        'hour': ['mean', 'std'],
        'day_of_week': ['mean'],
        'is_weekend': ['max'],
        'time_since_last_event': ['mean', 'std', 'max', 'sum'],
        'user_total_sessions': ['first'],
        'user_total_buys': ['first'],
        'user_avg_session_duration': ['first'],
        'prod_buy_ratio': ['mean', 'std']
    }
    session_df = df.groupby('user_session').agg(agg_funcs)
    session_df.columns = ['_'.join(col).strip() for col in session_df.columns.values]
    event_counts = df.groupby(['user_session', 'event_type']).size().unstack(fill_value=0)
    event_counts.columns = [f'count_{col.lower()}' for col in event_counts.columns]
    session_df = session_df.join(event_counts)
    session_df['buy_to_view_ratio'] = session_df.get('count_buy', 0) / (session_df.get('count_view', 0) + 1e-6)
    session_df['cart_to_view_ratio'] = session_df.get('count_add_cart', 0) / (session_df.get('count_view', 0) + 1e-6)
    session_df['remove_to_cart_ratio'] = session_df.get('count_remove_cart', 0) / (session_df.get('count_add_cart', 0) + 1e-6)
    return session_df

session_features = aggregate_session_features(combined_df)

train_sessions = train_df['user_session'].unique()
test_sessions = test_df['user_session'].unique()

train_final = session_features.loc[train_sessions].copy()
test_final = session_features.loc[test_sessions].copy()

y_train = np.log1p(train_df.groupby('user_session')['session_value'].first())
train_final['user_id_for_cv'] = train_df.groupby('user_session')['user_id'].first()

X_train = train_final.drop(columns=['user_id_for_cv'])
X_test = test_final.copy()

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    if c != 'user_id_for_cv':
        X_test[c] = 0
X_test = X_test[X_train.columns]

print("Özellik mühendisliği tamamlandı. Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("---------------------------------")
