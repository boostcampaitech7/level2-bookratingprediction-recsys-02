# preprocess.py
import pandas as pd
import yaml
import os
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineering import FeatureEngineer

def main():
    print("Starting preprocessing...")
    
    # 설정 파일 로드
    try:
        print("Loading config file...")
        with open('config/preprocessing_config.yaml') as f:
            config = yaml.safe_load(f)
        print("Config file loaded successfully!")
    except FileNotFoundError:
        print("Error: preprocessing_config.yaml not found in config directory!")
        return
    
    # 데이터 로드
    print("\nLoading data files...")
    try:
        books_df = pd.read_csv(config['data']['books_path'])
        print(f"Books data loaded: {books_df.shape[0]} rows")
        
        users_df = pd.read_csv(config['data']['users_path'])
        print(f"Users data loaded: {users_df.shape[0]} rows")
        
        train_ratings_df = pd.read_csv(config['data']['train_ratings_path'])
        print(f"Train ratings loaded: {train_ratings_df.shape[0]} rows")
        
        test_ratings_df = pd.read_csv(config['data']['test_ratings_path'])
        print(f"Test ratings loaded: {test_ratings_df.shape[0]} rows")
    except FileNotFoundError as e:
        print(f"Error: Data file not found! {str(e)}")
        return
    
    # 데이터 클리닝
    print("\nCleaning data...")
    cleaner = DataCleaner()
    
    print("Processing location data...")
    users_df = cleaner.clean_location(users_df)
    
    print("Processing age data...")
    users_df = cleaner.clean_age(users_df)
    
    print("Processing category data...")
    books_df = cleaner.clean_category(books_df)
    
    # 피처 엔지니어링
    print("\nCreating new features...")
    engineer = FeatureEngineer()
    
    print("Creating user features...")
    user_features = engineer.create_user_features(train_ratings_df)
    print(f"Created {user_features.shape[1]} user features")
    
    print("Creating book features...")
    book_features = engineer.create_book_features(train_ratings_df, books_df)
    print(f"Created {book_features.shape[1]} book features")
    
    # 저장 디렉토리 확인 및 생성
    save_path = config['features']['save_path']
    os.makedirs(save_path, exist_ok=True)
    
    # 전처리된 데이터 저장
    print("\nSaving processed features...")
    user_features.to_csv(f"{save_path}user_features.csv", index=False)
    print(f"User features saved to: {save_path}user_features.csv")
    
    book_features.to_csv(f"{save_path}book_features.csv", index=False)
    print(f"Book features saved to: {save_path}book_features.csv")
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()