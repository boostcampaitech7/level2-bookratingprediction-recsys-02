# preprocess_advanced.py
import pandas as pd
import numpy as np
import yaml
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.feature_processor import FeatureProcessor

def main():
    print("Starting advanced preprocessing...")
    
    # 1. 기존 피처 로드
    user_features = pd.read_csv('src/features/user_features.csv')
    book_features = pd.read_csv('src/features/book_features.csv')
    train_ratings = pd.read_csv('data/train_ratings.csv')
    
    # 2. 데이터 처리
    processor = FeatureProcessor()
    
    # 불균형 처리
    filtered_users, filtered_books = processor.handle_imbalance(
        user_features, book_features, 
        min_user_ratings=5, min_book_ratings=5
    )
    
    # 데이터 병합
    df = train_ratings.merge(filtered_users, on='user_id', how='inner')
    df = df.merge(filtered_books, on='isbn', how='inner')
    
    # 상호작용 피처 생성
    df = processor.create_interaction_features(df)
    
    # 피처 정규화
    df_processed = processor.normalize_features(df)
    
    # 결과 저장
    save_path = 'src/features/processed_features.csv'
    df_processed.to_csv(save_path, index=False)
    print(f"Processed features saved to: {save_path}")
    
    # 처리 결과 요약
    print("\nFeature Processing Summary:")
    print(f"Original samples: {len(train_ratings)}")
    print(f"Processed samples: {len(df_processed)}")
    print(f"Features created: {len(df_processed.columns)}")
    
    # 수치형 컬럼만 선택하여 상관관계 분석
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    correlation_matrix = df_processed[numeric_cols].corr()
    
    print("\nNumeric features for correlation analysis:")
    print(numeric_cols.tolist())
    
    correlation_matrix.to_csv('src/features/feature_correlations.csv')
    print("Correlation matrix saved to: src/features/feature_correlations.csv")
    
    # 피처별 기본 통계량 저장
    feature_stats = df_processed[numeric_cols].describe()
    feature_stats.to_csv('src/features/feature_statistics.csv')
    print("Feature statistics saved to: src/features/feature_statistics.csv")

if __name__ == "__main__":
    main()