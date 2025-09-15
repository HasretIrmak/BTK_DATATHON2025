import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lightgbm as lgb
import xgboost as xgb
import catboost as cbt

from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

import optuna

import os
from datetime import datetime

# Stil ayarları
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Uyarıları bastır
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = '/kaggle/input/datathon-2025'

train_df = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))
test_df = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))

print("Veri dosyaları başarıyla yüklendi.")


class EcommerceEDA:
    def __init__(self, train_path, test_path):
        """
        E-ticaret veri analizi sınıfı
        """
        self.train_df = pd.read_csv("/kaggle/input/datathon-2025/train.csv")
        self.test_df = pd.read_csv("/kaggle/input/datathon-2025/test.csv")
        self.combined_df = None

        print(f"Train veri boyutu: {self.train_df.shape}")
        print(f"Test veri boyutu: {self.test_df.shape}")

    def basic_info(self):
        """Temel veri bilgileri"""
        print("=" * 50)
        print("TEMEL VERİ BİLGİLERİ")
        print("=" * 50)

        print("\n📊 TRAIN VERİSİ:")
        print(f"Boyut: {self.train_df.shape}")
        print(f"Memory usage: {self.train_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print(f"\nData types:")
        print(self.train_df.dtypes.value_counts())

        print(f"\n📊 TEST VERİSİ:")
        print(f"Boyut: {self.test_df.shape}")

        print(f"\n🔍 EKSIK DEĞERLER:")
        train_null = self.train_df.isnull().sum()
        test_null = self.test_df.isnull().sum()

        print("Train:")
        print(train_null[train_null > 0] if train_null.sum() > 0 else "Eksik değer yok")
        print("\nTest:")
        print(test_null[test_null > 0] if test_null.sum() > 0 else "Eksik değer yok")

        print(f"\n📈 İLK 5 SATIR (TRAIN):")
        print(self.train_df.head())

        return self.train_df.describe(), self.test_df.describe()

    def target_analysis(self):
        """Session value hedef değişken analizi"""
        print("\n" + "=" * 50)
        print("HEDEF DEĞİŞKEN ANALİZİ (SESSION_VALUE)")
        print("=" * 50)

        target = self.train_df['session_value']

        print(f"📊 Temel istatistikler:")
        print(f"Ortalama: {target.mean():.2f}")
        print(f"Medyan: {target.median():.2f}")
        print(f"Standart sapma: {target.std():.2f}")
        print(f"Min: {target.min():.2f}")
        print(f"Max: {target.max():.2f}")
        print(f"Skewness: {target.skew():.2f}")
        print(f"Kurtosis: {target.kurtosis():.2f}")

        # Aykırı değer tespiti
        Q1 = target.quantile(0.25)
        Q3 = target.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = target[(target < lower_bound) | (target > upper_bound)]
        print(f"\n🚨 Aykırı değerler: {len(outliers)} adet ({len(outliers) / len(target) * 100:.2f}%)")

        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        axes[0, 0].hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(target.mean(), color='red', linestyle='--', label=f'Ortalama: {target.mean():.2f}')
        axes[0, 0].axvline(target.median(), color='green', linestyle='--', label=f'Medyan: {target.median():.2f}')
        axes[0, 0].set_title('Session Value Dağılımı')
        axes[0, 0].set_xlabel('Session Value')
        axes[0, 0].set_ylabel('Frekans')
        axes[0, 0].legend()

        # Log dönüşümlü histogram
        log_target = np.log1p(target)
        axes[0, 1].hist(log_target, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Log Dönüşümlü Session Value Dağılımı')
        axes[0, 1].set_xlabel('Log(Session Value + 1)')
        axes[0, 1].set_ylabel('Frekans')

        # Box plot
        axes[1, 0].boxplot(target, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightcoral'))
        axes[1, 0].set_title('Session Value Box Plot')
        axes[1, 0].set_ylabel('Session Value')

        # QQ plot normallik testi
        stats.probplot(target, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normallik Testi)')

        plt.tight_layout()
        plt.show()

        return target.describe()

    def categorical_analysis(self):
        """Kategorik değişkenler analizi"""
        print("\n" + "=" * 50)
        print("KATEGORİK DEĞİŞKENLER ANALİZİ")
        print("=" * 50)

        categorical_cols = ['event_type', 'product_id', 'category_id', 'user_id', 'user_session']

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Train verisi için sayım
                value_counts = self.train_df[col].value_counts().head(20)

                print(f"\n📊 {col.upper()}:")
                print(f"Unique değer sayısı (train): {self.train_df[col].nunique()}")
                if col in self.test_df.columns:
                    print(f"Unique değer sayısı (test): {self.test_df[col].nunique()}")

                print(f"En sık görülen 5 değer:")
                print(value_counts.head())

                # Görselleştirme
                if len(value_counts) <= 10:
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                else:
                    axes[i].bar(range(len(value_counts)), value_counts.values)
                    axes[i].set_xlabel(f'Top 20 {col}')

                axes[i].set_title(f'{col} Dağılımı')
                axes[i].set_ylabel('Frekans')

        plt.tight_layout()
        plt.show()

    def event_type_analysis(self):
        """Event type detaylı analizi"""
        print("\n" + "=" * 50)
        print("EVENT TYPE DETAYLI ANALİZİ")
        print("=" * 50)

        # Event type ile session value ilişkisi
        event_stats = self.train_df.groupby('event_type')['session_value'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        print("📊 Event Type bazında Session Value istatistikleri:")
        print(event_stats)

        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Event type dağılımı
        event_counts = self.train_df['event_type'].value_counts()
        axes[0, 0].pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Event Type Dağılımı')

        # Event type vs session value (box plot)
        self.train_df.boxplot(column='session_value', by='event_type', ax=axes[0, 1])
        axes[0, 1].set_title('Event Type vs Session Value')
        axes[0, 1].set_xlabel('Event Type')

        # Event type vs session value (violin plot)
        sns.violinplot(data=self.train_df, x='event_type', y='session_value', ax=axes[1, 0])
        axes[1, 0].set_title('Session Value Dağılımı (Event Type)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Ortalama session value karşılaştırması
        mean_values = self.train_df.groupby('event_type')['session_value'].mean().sort_values(ascending=False)
        axes[1, 1].bar(mean_values.index, mean_values.values, color='coral')
        axes[1, 1].set_title('Event Type Ortalama Session Value')
        axes[1, 1].set_ylabel('Ortalama Session Value')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

        return event_stats

    def user_session_analysis(self):
        """Kullanıcı ve session analizi"""
        print("\n" + "=" * 50)
        print("KULLANICI VE SESSION ANALİZİ")
        print("=" * 50)

        # Session başına event sayısı
        session_event_count = self.train_df.groupby('user_session').size().reset_index(name='event_count')
        session_value_map = self.train_df.groupby('user_session')['session_value'].first().reset_index()
        session_stats = session_event_count.merge(session_value_map, on='user_session')

        print(f"📊 Session istatistikleri:")
        print(f"Toplam unique session sayısı: {self.train_df['user_session'].nunique()}")
        print(f"Session başına ortalama event sayısı: {session_stats['event_count'].mean():.2f}")
        print(f"Session başına medyan event sayısı: {session_stats['event_count'].median():.2f}")

        # Kullanıcı başına session sayısı
        user_session_count = self.train_df.groupby('user_id')['user_session'].nunique().reset_index(
            name='session_count')

        print(f"\n👤 Kullanıcı istatistikleri:")
        print(f"Toplam unique kullanıcı sayısı: {self.train_df['user_id'].nunique()}")
        print(f"Kullanıcı başına ortalama session sayısı: {user_session_count['session_count'].mean():.2f}")

        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Session başına event sayısı dağılımı
        axes[0, 0].hist(session_stats['event_count'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 0].set_title('Session Başına Event Sayısı Dağılımı')
        axes[0, 0].set_xlabel('Event Sayısı')
        axes[0, 0].set_ylabel('Frekans')

        # Event sayısı vs session value
        axes[0, 1].scatter(session_stats['event_count'], session_stats['session_value'], alpha=0.6, s=1)
        axes[0, 1].set_title('Event Sayısı vs Session Value')
        axes[0, 1].set_xlabel('Event Sayısı')
        axes[0, 1].set_ylabel('Session Value')

        # Korelasyon hesaplama
        correlation = session_stats['event_count'].corr(session_stats['session_value'])
        axes[0, 1].text(0.05, 0.95, f'Korelasyon: {correlation:.3f}',
                        transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

        # Kullanıcı başına session sayısı
        axes[1, 0].hist(user_session_count['session_count'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Kullanıcı Başına Session Sayısı')
        axes[1, 0].set_xlabel('Session Sayısı')
        axes[1, 0].set_ylabel('Kullanıcı Sayısı')

        # Session value dağılımı (log scale)
        axes[1, 1].hist(session_stats['session_value'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Session Value Dağılımı (Log Scale)')
        axes[1, 1].set_xlabel('Session Value')
        axes[1, 1].set_ylabel('Frekans (Log)')

        plt.tight_layout()
        plt.show()

        return session_stats, user_session_count

    def product_category_analysis(self):
        """Ürün ve kategori analizi"""
        print("\n" + "=" * 50)
        print("ÜRÜN VE KATEGORİ ANALİZİ")
        print("=" * 50)

        # Kategori bazında analiz
        category_stats = self.train_df.groupby('category_id').agg({
            'session_value': ['mean', 'count', 'std'],
            'product_id': 'nunique'
        }).round(2)
        category_stats.columns = ['avg_session_value', 'session_count', 'session_std', 'unique_products']
        category_stats = category_stats.reset_index().sort_values('avg_session_value', ascending=False)

        print(f"📊 Kategori istatistikleri:")
        print(f"Toplam kategori sayısı: {self.train_df['category_id'].nunique()}")
        print(f"Toplam ürün sayısı: {self.train_df['product_id'].nunique()}")
        print(f"\nEn yüksek ortalama session value'ya sahip ilk 10 kategori:")
        print(category_stats.head(10))

        # Ürün bazında analiz
        product_stats = self.train_df.groupby('product_id').agg({
            'session_value': ['mean', 'count'],
            'category_id': 'first'
        }).round(2)
        product_stats.columns = ['avg_session_value', 'interaction_count', 'category_id']
        product_stats = product_stats.reset_index()

        # En popüler ürünler
        popular_products = product_stats.nlargest(10, 'interaction_count')
        print(f"\n🔥 En popüler 10 ürün:")
        print(popular_products)

        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Top 20 kategori - ortalama session value
        top_categories = category_stats.head(20)
        axes[0, 0].barh(range(len(top_categories)), top_categories['avg_session_value'])
        axes[0, 0].set_yticks(range(len(top_categories)))
        axes[0, 0].set_yticklabels(top_categories['category_id'])
        axes[0, 0].set_title('Top 20 Kategori - Ortalama Session Value')
        axes[0, 0].set_xlabel('Ortalama Session Value')

        # Kategori başına unique ürün sayısı
        axes[0, 1].scatter(category_stats['unique_products'], category_stats['avg_session_value'], alpha=0.6)
        axes[0, 1].set_title('Kategori Başına Unique Ürün vs Ortalama Session Value')
        axes[0, 1].set_xlabel('Unique Ürün Sayısı')
        axes[0, 1].set_ylabel('Ortalama Session Value')

        # Ürün popülerliği dağılımı
        axes[1, 0].hist(product_stats['interaction_count'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_title('Ürün Etkileşim Sayısı Dağılımı')
        axes[1, 0].set_xlabel('Etkileşim Sayısı')
        axes[1, 0].set_ylabel('Ürün Sayısı')
        axes[1, 0].set_yscale('log')

        # Session value vs interaction count (ürün bazında)
        axes[1, 1].scatter(product_stats['interaction_count'], product_stats['avg_session_value'], alpha=0.5, s=1)
        axes[1, 1].set_title('Ürün Etkileşim Sayısı vs Ortalama Session Value')
        axes[1, 1].set_xlabel('Etkileşim Sayısı')
        axes[1, 1].set_ylabel('Ortalama Session Value')
        axes[1, 1].set_xscale('log')

        plt.tight_layout()
        plt.show()

        return category_stats, product_stats

    def correlation_analysis(self):
        """Korelasyon analizi"""
        print("\n" + "=" * 50)
        print("KORELASYON ANALİZİ")
        print("=" * 50)

        # Basit numerik özellikler için korelasyon analizi
        session_event_count = self.train_df.groupby('user_session').size().reset_index(name='event_count')
        session_value_map = self.train_df.groupby('user_session')['session_value'].first().reset_index()
        session_stats = session_event_count.merge(session_value_map, on='user_session')

        # Event type sayılarını hesapla
        event_type_counts = pd.crosstab(self.train_df['user_session'], self.train_df['event_type'])
        session_analysis = session_stats.merge(event_type_counts, left_on='user_session', right_index=True)

        # Kategori ve ürün çeşitliliği
        session_diversity = self.train_df.groupby('user_session').agg({
            'product_id': 'nunique',
            'category_id': 'nunique'
        }).reset_index()
        session_diversity.columns = ['user_session', 'unique_products', 'unique_categories']

        # Final dataset
        final_analysis = session_analysis.merge(session_diversity, on='user_session')

        # Korelasyon matrisi
        numeric_cols = final_analysis.select_dtypes(include=[np.number]).columns
        correlation_matrix = final_analysis[numeric_cols].corr()

        # Session value ile korelasyonlar
        target_corr = correlation_matrix['session_value'].abs().sort_values(ascending=False)
        print(f"📊 Session Value ile korelasyonlar:")
        print(target_corr)

        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Korelasyon heatmap
        mask = np.triu(correlation_matrix)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=axes[0, 0])
        axes[0, 0].set_title('Özellik Korelasyon Matrisi')

        # Session value korelasyon barplot
        target_corr_filtered = target_corr[target_corr.index != 'session_value']
        axes[0, 1].barh(range(len(target_corr_filtered)), target_corr_filtered.values)
        axes[0, 1].set_yticks(range(len(target_corr_filtered)))
        axes[0, 1].set_yticklabels(target_corr_filtered.index)
        axes[0, 1].set_title('Session Value ile Korelasyonlar')
        axes[0, 1].set_xlabel('Mutlak Korelasyon')

        # Event count vs session value scatter
        axes[1, 0].scatter(final_analysis['event_count'], final_analysis['session_value'], alpha=0.6)
        axes[1, 0].set_xlabel('Event Count')
        axes[1, 0].set_ylabel('Session Value')
        axes[1, 0].set_title('Event Count vs Session Value')

        # Unique products vs session value
        axes[1, 1].scatter(final_analysis['unique_products'], final_analysis['session_value'], alpha=0.6)
        axes[1, 1].set_xlabel('Unique Products')
        axes[1, 1].set_ylabel('Session Value')
        axes[1, 1].set_title('Unique Products vs Session Value')

        plt.tight_layout()
        plt.show()

        return correlation_matrix

    def advanced_visualization(self):
        """Gelişmiş görselleştirmeler"""
        print("\n" + "=" * 50)
        print("GELİŞMİŞ GÖRSELLEŞTİRMELER")
        print("=" * 50)

        # Session value'nun event type'lara göre dağılımı (interaktif)
        fig = px.box(self.train_df, x='event_type', y='session_value',
                     title='Session Value Dağılımı (Event Type Bazında)')
        fig.show()

        # Kategori bazında session value heatmap
        category_event = pd.crosstab(self.train_df['category_id'], self.train_df['event_type'])

        plt.figure(figsize=(12, 8))
        sns.heatmap(category_event.head(20), annot=True, cmap='YlOrRd', fmt='d')
        plt.title('Top 20 Kategori - Event Type Heatmap')
        plt.ylabel('Category ID')
        plt.xlabel('Event Type')
        plt.tight_layout()
        plt.show()

    def data_quality_check(self):
        """Veri kalitesi kontrolü"""
        print("\n" + "=" * 50)
        print("VERİ KALİTESİ KONTROLÜ")
        print("=" * 50)

        # Duplicate kontrolü
        train_duplicates = self.train_df.duplicated().sum()
        test_duplicates = self.test_df.duplicated().sum()

        print(f"🔍 Duplicate satırlar:")
        print(f"Train: {train_duplicates}")
        print(f"Test: {test_duplicates}")

        # Train ve test arasındaki farklar
        print(f"\n📊 Train vs Test karşılaştırması:")

        for col in ['event_type', 'category_id']:
            train_unique = set(self.train_df[col].unique())
            test_unique = set(self.test_df[col].unique())

            print(f"\n{col}:")
            print(f"  Train unique: {len(train_unique)}")
            print(f"  Test unique: {len(test_unique)}")
            print(f"  Train'de olup test'te olmayan: {len(train_unique - test_unique)}")
            print(f"  Test'te olup train'de olmayan: {len(test_unique - train_unique)}")

    def run_full_analysis(self):
        """Tam analizi çalıştır"""
        print("🚀 E-TİCARET SESSION VALUE ANALİZİ BAŞLIYOR")
        print("=" * 60)

        # Temel bilgiler
        basic_stats = self.basic_info()

        # Hedef değişken analizi
        target_stats = self.target_analysis()

        # Kategorik değişkenler
        self.categorical_analysis()

        # Event type analizi
        event_stats = self.event_type_analysis()

        # Session ve kullanıcı analizi
        session_stats, user_stats = self.user_session_analysis()

        # Ürün ve kategori analizi
        category_stats, product_stats = self.product_category_analysis()

        # Korelasyon analizi
        corr_matrix = self.correlation_analysis()

        # Gelişmiş görselleştirmeler
        self.advanced_visualization()

        # Veri kalitesi
        self.data_quality_check()

        print("\n✅ ANALİZ TAMAMLANDI!")

        return {
            'basic_stats': basic_stats,
            'target_stats': target_stats,
            'event_stats': event_stats,
            'session_stats': session_stats,
            'user_stats': user_stats,
            'category_stats': category_stats,
            'product_stats': product_stats,
            'correlation_matrix': corr_matrix
        }


# Kullanım örneği:
if __name__ == "__main__":
    # EDA sınıfını başlat
    eda = EcommerceEDA('train.csv', 'test.csv')

    # Tam analizi çalıştır
    results = eda.run_full_analysis()
