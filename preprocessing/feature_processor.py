# src/preprocessing/feature_processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def handle_imbalance(self, user_features, book_features, min_user_ratings=5, min_book_ratings=5):
        """
        평점 수가 적은 사용자와 책 필터링
        """
        print("Handling data imbalance...")
        
        # 사용자 필터링
        filtered_users = user_features[user_features['user_rating_count'] >= min_user_ratings]
        print(f"Filtered users: {len(filtered_users)} / {len(user_features)}")
        
        # 책 필터링
        filtered_books = book_features[book_features['book_rating_count'] >= min_book_ratings]
        print(f"Filtered books: {len(filtered_books)} / {len(book_features)}")
        
        return filtered_users, filtered_books
    
    def create_interaction_features(self, df):
        """
        사용자-책 상호작용 피처 생성
        """
        print("Creating interaction features...")
        
        # 평균 평점 차이
        df['user_book_rating_diff'] = df['user_mean_rating'] - df['book_mean_rating']
        
        # 평점 수 비율
        df['rating_count_ratio'] = df['user_rating_count'] / df['book_rating_count']
        
        # 평점 표준편차 비율
        df['rating_std_ratio'] = df['user_rating_std'] / df['book_rating_std'].replace(0, 1)
        
        print(f"Created features: {['user_book_rating_diff', 'rating_count_ratio', 'rating_std_ratio']}")
        return df
    
    def normalize_features(self, df, features_to_scale=None):
        """
        피처 정규화
        """
        print("Normalizing features...")
        
        if features_to_scale is None:
            features_to_scale = [
                'user_mean_rating', 'book_mean_rating',
                'user_rating_std', 'book_rating_std',
                'user_rating_count', 'book_rating_count'
            ]
        
        # 기존 데이터 백업
        df_scaled = df.copy()
        
        # 정규화 적용
        df_scaled[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        
        print(f"Normalized features: {features_to_scale}")
        return df_scaled