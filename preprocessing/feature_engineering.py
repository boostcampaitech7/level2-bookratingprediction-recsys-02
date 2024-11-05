# src/preprocessing/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def preprocess_features(books_df, users_df, ratings_df, min_rating_threshold=5):
    """
    DeepFM을 위한 전처리 및 피처 엔지니어링
    """
    
    # 1. 이상치 처리
    books_df = books_df[
        (books_df['year_of_publication'] >= 1900) & 
        (books_df['year_of_publication'] <= 2006)
    ].copy()
    
    # 2. 출판사 카테고리화
    def categorize_publisher(publisher):
        publisher = str(publisher).lower()
        if any(x in publisher for x in ['harlequin', 'silhouette', 'avon', 'mira']):
            return 'Romance'
        elif any(x in publisher for x in ['scholastic', 'puffin', 'golden books']):
            return 'Children/Young Adult'
        elif any(x in publisher for x in ['oxford', 'academic', 'university press']):
            return 'Academic'
        elif any(x in publisher for x in ['tor books', 'del rey', 'baen']):
            return 'SF/Fantasy'
        elif any(x in publisher for x in ['penguin', 'random house', 'harpercollins']):
            return 'Major Publisher'
        elif any(x in publisher for x in ['comics', 'manga', 'tokyopop']):
            return 'Comics/Manga'
        return 'Others'

    def calculate_rating_reliability(count):
        if count < 5:
            return 0.5
        elif count < 10:
            return 0.7
        elif count < 50:
            return 0.9
        else:
            return 1.0
    
    books_df['publisher_category'] = books_df['publisher'].apply(categorize_publisher)
    
    # 3. 사용자 통계 피처 확장
    user_stats = ratings_df.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max']
    }).reset_index()
    user_stats.columns = ['user_id', 'user_rating_count', 'user_mean_rating', 
                         'user_rating_std', 'user_min_rating', 'user_max_rating']
    
    user_stats['user_rating_range'] = user_stats['user_max_rating'] - user_stats['user_min_rating']
    user_stats['user_rating_reliability'] = user_stats['user_rating_count'].apply(calculate_rating_reliability)
    
    # 최소 평점 개수 필터링
    active_users = user_stats[user_stats['user_rating_count'] >= min_rating_threshold]['user_id']
    ratings_df = ratings_df[ratings_df['user_id'].isin(active_users)]
    
    # 4. 책 통계 피처 확장
    book_stats = ratings_df.groupby('isbn').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max']
    }).reset_index()
    book_stats.columns = ['isbn', 'book_rating_count', 'book_mean_rating', 
                         'book_rating_std', 'book_min_rating', 'book_max_rating']
    
    book_stats['book_rating_range'] = book_stats['book_max_rating'] - book_stats['book_min_rating']
    book_stats['book_rating_reliability'] = book_stats['book_rating_count'].apply(calculate_rating_reliability)
    
    # 5. 연도 그룹화
    def get_year_group(year):
        if year < 1900 or year > 2006:
            return 'others'
        elif year < 1950:
            return 'old'
        elif year < 1980:
            return 'medium'
        elif year < 2000:
            return 'recent'
        else:
            return 'new'
    
    books_df['year_group'] = books_df['year_of_publication'].apply(get_year_group)
    
    # 6. 연속형/범주형 피처 구분
    continuous_features = [
        'user_rating_count', 'user_mean_rating', 'user_rating_std',
        'user_rating_range', 'user_rating_reliability',
        'book_rating_count', 'book_mean_rating', 'book_rating_std',
        'book_rating_range', 'book_rating_reliability'
    ]
    
    categorical_features = [
        'publisher_category',
        'year_group'
    ]
    
    # 7. 데이터 병합
    final_df = ratings_df.merge(books_df[['isbn', 'publisher_category', 'year_group']], on='isbn')
    final_df = final_df.merge(user_stats, on='user_id')
    final_df = final_df.merge(book_stats, on='isbn')
    
    # 8. 상호작용 피처 추가
    final_df['user_book_rating_diff'] = final_df['user_mean_rating'] - final_df['book_mean_rating']
    final_df['rating_count_ratio'] = np.log1p(final_df['user_rating_count']) - np.log1p(final_df['book_rating_count'])
    final_df['rating_std_ratio'] = final_df['user_rating_std'] / (final_df['book_rating_std'] + 1e-6)
    final_df['rating_reliability_interaction'] = final_df['user_rating_reliability'] * final_df['book_rating_reliability']
    final_df['rating_range_interaction'] = final_df['user_rating_range'] * final_df['book_rating_range']
    
    # 상호작용 피처 추가
    continuous_features.extend([
        'user_book_rating_diff', 'rating_count_ratio', 'rating_std_ratio',
        'rating_reliability_interaction', 'rating_range_interaction'
    ])
    
    # 9. 범주형 변수 인코딩
    le_dict = {}
    for feat in categorical_features:
        le = LabelEncoder()
        final_df[feat] = le.fit_transform(final_df[feat].astype(str))
        le_dict[feat] = le
    
    # 10. 연속형 변수 정규화
    for feat in continuous_features:
        mean = final_df[feat].mean()
        std = final_df[feat].std()
        final_df[feat] = (final_df[feat] - mean) / std
    
    # 11. DeepFM 입력 형식으로 변환
    categorical_values = final_df[categorical_features].values
    continuous_values = final_df[continuous_features].values
    
    # 필드 차원 정보
    field_dims = []
    field_names = []
    
    # 범주형 변수의 차원
    for feat in categorical_features:
        field_dims.append(len(le_dict[feat].classes_))
        field_names.append(feat)
    
    # 연속형 변수의 차원 (1로 설정)
    for feat in continuous_features:
        field_dims.append(1)
        field_names.append(feat)
    
    return {
        'categorical_values': categorical_values,
        'continuous_values': continuous_values,
        'field_dims': field_dims,
        'field_names': field_names,
        'continuous_cols': continuous_features,
        'categorical_cols': categorical_features,
        'label_encoders': le_dict,
        'final_df': final_df
    }

def test_preprocessing(data_dict):
    """
    전처리 결과 검증
    """
    print("Field dimensions:", data_dict['field_dims'])
    print("\nField names:", data_dict['field_names'])
    print("\nCategorical features shape:", data_dict['categorical_values'].shape)
    print("Continuous features shape:", data_dict['continuous_values'].shape)
    print("\nSample of final dataframe:")
    print(data_dict['final_df'].head())