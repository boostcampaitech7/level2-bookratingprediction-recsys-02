# src/preprocessing/data_cleaner.py
import pandas as pd
import numpy as np

class DataCleaner:
    def __init__(self):
        pass
    
    def clean_location(self, df):
        """
        location 컬럼을 city, state, country로 분리
        """
        print("Cleaning location data...")
        
        def split_location(x):
            parts = str(x).split(',')
            parts = [p.strip() for p in parts]
            
            if len(parts) >= 3:
                return pd.Series({'city': parts[0], 'state': parts[1], 'country': parts[2]})
            else:
                return pd.Series({'city': 'unknown', 'state': 'unknown', 'country': 'unknown'})
        
        location_df = df['location'].apply(split_location)
        print(f"Created {len(location_df.columns)} location features")
        return pd.concat([df, location_df], axis=1)
    
    def clean_age(self, df):
        """
        나이 결측치 처리 및 구간화
        """
        print("Cleaning age data...")
        
        # 중앙값으로 결측치 대체
        df['age'] = df['age'].fillna(df['age'].median())
        
        # 나이 구간화
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 20, 30, 40, 50, 60, 100],
                                labels=['0-20', '21-30', '31-40', '41-50', '51-60', '60+'])
        
        print(f"Age groups created: {df['age_group'].nunique()} groups")
        return df
    
    def clean_category(self, df):
        """
        카테고리 전처리
        """
        print("Cleaning category data...")
        
        def extract_categories(x):
            if pd.isna(x):
                return ['unknown']
            try:
                categories = eval(x)
                return [cat.strip() for cat in categories]
            except:
                return ['unknown']
        
        df['categories'] = df['category'].apply(extract_categories)
        print(f"Processed {df['categories'].notna().sum()} category entries")
        return df