print("Katmanlı Ensemble model eğitimi başlıyor...")

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)
groups = train_final['user_id_for_cv']
oof_preds = np.zeros((len(X_train), 3))  # LGB, XGB, CAT
test_preds = np.zeros((len(X_test), 3))

# --- LightGBM Optuna Optimizasyonu ---
def objective_lgb(trial):
    lgb_params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    mse_scores = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
            y_pred = model.predict(X_va)
            mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
    return np.mean(mse_scores)

study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=20)
lgb_params = study_lgb.best_params
lgb_params.update({'objective': 'regression_l1', 'metric': 'rmse', 'verbose': -1, 'n_jobs': -1, 'seed': 42})
print("LightGBM için en iyi parametreler:", lgb_params)

# LightGBM Eğitimi
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    if len(X_va) > 0:
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_preds[va_idx, 0] = model.predict(X_va)
        test_preds[:, 0] += model.predict(X_test) / n_splits
    else:
        print(f"Fold {fold}: Boş doğrulama seti, LGBM eğitimi atlanıyor.")
print("LGBM eğitimi tamamlandı.")


# --- XGBoost Optuna Optimizasyonu ---
def objective_xgb(trial):
    xgb_params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 100
    }
    mse_scores = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            y_pred = model.predict(X_va)
            mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
    return np.mean(mse_scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)
xgb_params = study_xgb.best_params
xgb_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 100})
print("XGBoost için en iyi parametreler:", xgb_params)


# XGBoost Eğitimi
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    if len(X_va) > 0:
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False
        )
        oof_preds[va_idx, 1] = model.predict(X_va)
        test_preds[:, 1] += model.predict(X_test) / n_splits
    else:
        print(f"Fold {fold}: Boş doğrulama seti, XGBoost eğitimi atlanıyor.")
print("XGBoost eğitimi tamamlandı.")

# --- CatBoost Optuna Optimizasyonu ---
def objective_cbt(trial):
    cbt_params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.0, 10.0),
        'loss_function': 'RMSE',
        'verbose': 0,
        'random_seed': 42
    }
    mse_scores = []
    for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        if len(X_va) > 0:
            model = cbt.CatBoostRegressor(**cbt_params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100)
            y_pred = model.predict(X_va)
            mse_scores.append(mean_squared_error(np.expm1(y_va), np.expm1(y_pred)))
    return np.mean(mse_scores)

study_cbt = optuna.create_study(direction='minimize')
study_cbt.optimize(objective_cbt, n_trials=20)
cbt_params = study_cbt.best_params
cbt_params.update({'loss_function': 'RMSE', 'verbose': 0, 'random_seed': 42})
print("CatBoost için en iyi parametreler:", cbt_params)



# CatBoost Eğitimi
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y_train[tr_idx], y_train[va_idx]
    if len(X_va) > 0:
        model = cbt.CatBoostRegressor(**cbt_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=100)
        oof_preds[va_idx, 2] = model.predict(X_va)
        test_preds[:, 2] += model.predict(X_test) / n_splits
    else:
        print(f"Fold {fold}: Boş doğrulama seti, CatBoost eğitimi atlanıyor.")
print("CatBoost eğitimi tamamlandı.")


# MSE Sonucu
oof_mse = mean_squared_error(np.expm1(y_train), np.expm1(oof_preds.mean(axis=1)))
print(f"Katman 0 OOF Ortalaması MSE: {oof_mse:.4f}")

meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(oof_preds, y_train)

final_preds_log = meta_model.predict(test_preds)
final_preds = np.expm1(final_preds_log)

print("Katmanlı Ensemble eğitimi tamamlandı.")

# Submission DataFrame oluştur
submission_df = pd.DataFrame({
    'user_session': X_test.index,
    'session_value': final_preds
})

print("📊 Submission DataFrame oluşturuldu!")
print(f"Shape: {submission_df.shape}")

# KAGGLE WORKING DIRECTORY'E KAYDETME

submission_path_desc = '/kaggle/working/submission_31_XG_Last.csv'
submission_df.to_csv(submission_path_desc, index=False)
print(f"✅ Açıklayıcı dosya kaydedildi: {submission_path_desc}")

# DOSYA DOĞRULAMA

print("\n🔍 DOSYA DOĞRULAMA:")
print("-" * 40)

# Ana dosya seç (timestamp'li olanı)
main_submission = submission_path_desc

# Dosya boyutu
if os.path.exists(main_submission):
    file_size = os.path.getsize(main_submission)
    print(f"📁 Dosya boyutu: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
else:
    print("❌ Dosya bulunamadı!")

# DataFrame bilgileri
print(f"📊 Satır sayısı: {submission_df.shape[0]:,}")
print(f"📊 Sütun sayısı: {submission_df.shape[1]}")
print(f"📋 Sütunlar: {list(submission_df.columns)}")

# İlk ve son satırlar
print(f"\n📋 İLK 5 SATIR:")
print(submission_df.head())

print(f"\n📋 SON 5 SATIR:")
print(submission_df.tail())

# İSTATİSTİKSEL KONTROL

print(f"\n📈 SESSION_VALUE İSTATİSTİKLERİ:")
print(f"   Count: {submission_df['session_value'].count():,}")
print(f"   Mean: {submission_df['session_value'].mean():.2f}")
print(f"   Median: {submission_df['session_value'].median():.2f}")
print(f"   Std: {submission_df['session_value'].std():.2f}")
print(f"   Min: {submission_df['session_value'].min():.2f}")
print(f"   Max: {submission_df['session_value'].max():.2f}")
print(f"   25th percentile: {submission_df['session_value'].quantile(0.25):.2f}")
print(f"   75th percentile: {submission_df['session_value'].quantile(0.75):.2f}")

# =============================================================================
# HATA KONTROLLERI
# =============================================================================

print(f"\n🔍 HATA KONTROLLERI:")
print("-" * 40)

# Missing values
missing = submission_df.isnull().sum().sum()
if missing > 0:
    print(f"⚠️ Missing values: {missing}")
else:
    print("✅ Missing values: 0")

# Infinite values
infinite = np.isinf(submission_df['session_value']).sum()
if infinite > 0:
    print(f"⚠️ Infinite values: {infinite}")
else:
    print("✅ Infinite values: 0")

# Negative values
negative = (submission_df['session_value'] < 0).sum()
if negative > 0:
    print(f"⚠️ Negative values: {negative}")
else:
    print("✅ Negative values: 0")

# Duplicate user_sessions
duplicates = submission_df['user_session'].duplicated().sum()
if duplicates > 0:
    print(f"⚠️ Duplicate sessions: {duplicates}")
else:
    print("✅ Duplicate sessions: 0")

# Data types
print(f"\n📋 DATA TYPES:")
print(f"   user_session: {submission_df['user_session'].dtype}")
print(f"   session_value: {submission_df['session_value'].dtype}")

# =============================================================================
# KAGGLE FORMAT KONTROL
# =============================================================================

print(f"\n🎯 KAGGLE FORMAT KONTROL:")
print("-" * 40)

# Gerekli sütunlar
required_cols = ['user_session', 'session_value']
has_all_cols = all(col in submission_df.columns for col in required_cols)
if has_all_cols:
    print("✅ Tüm gerekli sütunlar mevcut")
else:
    print("❌ Eksik sütunlar var!")

# Sütun sırası
if list(submission_df.columns) == required_cols:
    print("✅ Sütun sırası doğru")
else:
    print("⚠️ Sütun sırası farklı")

# =============================================================================
# SON MESAJ
# =============================================================================

print(f"\n" + "=" * 60)
print("🎉 KAGGLE SUBMISSION HAZIR!")
print("=" * 60)
print(f"📁 Ana dosya: {main_submission}")
print(f"📈 Ortalama tahmin: {submission_df['session_value'].mean():.2f}")
print(f"📏 Min-Max: {submission_df['session_value'].min():.2f} - {submission_df['session_value'].max():.2f}")
print(f"🎯 Satır sayısı: {len(submission_df):,}")
print(f"\n🚀 Dosya '/kaggle/working/' klasöründe hazır!")
print(f"💡 Notebook'u 'Save Version' yaparak dosyaları indirin!")
print("=" * 60)

# Working directory içeriğini listele
print(f"\n📁 /kaggle/working/ KLASÖR İÇERİĞİ:")
print("-" * 40)
try:
    working_files = os.listdir('/kaggle/working/')
    csv_files = [f for f in working_files if f.endswith('.csv')]

    print(f"Toplam CSV dosyası: {len(csv_files)}")
    for csv_file in csv_files:
        file_path = f'/kaggle/working/{csv_file}'
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  📄 {csv_file} ({size_kb:.1f} KB)")

except Exception as e:
    print(f"Klasör listelenemedi: {e}")

print(f"\n✅ Submission tamamlandı! Good luck! 🍀")
