import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    # for i in range(len(res)-1, 0, -1):
    #     if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
    #         res.pop(i)

    return res



def cluster_categories(categories, similarity_threshold=0.7):
    """
    BERT를 사용하여 유사한 카테고리를 군집화하는 함수
    """
    # NaN 값 제거 및 문자열로 변환
    categories = [str(cat) for cat in categories if pd.notna(cat)]
    
    # BERT 모델 로드
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    # 카테고리 임베딩 생성
    category_embeddings = model.encode(categories)
    
    # 계층적 군집화 수행
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=(1 - similarity_threshold),
        metric='cosine',
        linkage='average'
    ).fit(category_embeddings)
    
    # 군집별로 카테고리 그룹화
    clusters = defaultdict(list)
    for category, label in zip(categories, clustering.labels_):
        clusters[label].append(category)
    
    # 각 군집에서 대표 카테고리 선정
    mapping = {}
    for cluster in clusters.values():
        # 가장 짧은 이름을 대표값으로 선택
        representative = min(cluster, key=len)
        for category in cluster:
            mapping[category] = representative
            
    return mapping

def apply_category_clustering(books, column, threshold=0.7):
    """
    데이터프레임의 카테고리를 군집화하여 적용하는 함수
    """
    books_ = books.copy()
    # NaN이 아닌 유니크한 카테고리 목록 추출
    unique_categories = books_[column].dropna().unique().tolist()
    
    # 카테고리 군집화
    category_mapping = cluster_categories(unique_categories, threshold)
    
    # 새로운 카테고리 컬럼 생성 (NaN 값은 그대로 유지)
    books_[f'{column}_clustered'] = books_[column].apply(
        lambda x: category_mapping.get(str(x), x) if pd.notna(x) else x
    )
    
    return books_

def process_context_data(users, books):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()

    # 데이터 전처리 (전처리는 각자의 상황에 맞게 진행해주세요!)
    # language 결측치 보간
    books_['isbn_code'] = books_.isbn.apply(lambda x : x[:2])
    code_book = books_[['isbn_code', 'language']].drop_duplicates()
    code_book = code_book.groupby('isbn_code').value_counts()
    code_book = code_book[code_book==1] # 1개 언어로 매칭되는 경우
    code_dict = {isbn: codes for isbn, codes in code_book.index}
    books_.loc[books_['language'].isna(), 'language'] = books_.loc[books_['language'].isna(), 'isbn_code'].map(code_dict)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    
    # category 통합
    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0].replace('\'','').replace('\"','').lower() if not pd.isna(x) else np.nan)
    books_ = apply_category_clustering(books_, 'category', 0.7)
    books_ = apply_category_clustering(books_, 'category_clustered', 0.7)
    etc_lst = books_.groupby('category_clustered_clustered')['category'].nunique()[books_.groupby('category_clustered_clustered')['category'].nunique() == 1].index
    books_['category'] = np.where(books_.category_clustered_clustered.isin(etc_lst), 'etc', books_.category_clustered_clustered)
    
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    books_['book_title_len'] = [len(title) for title in books_['book_title']]
    books_['book_author'] = books_['book_author'].fillna('unknown')
    
    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)  # 10대, 20대, 30대, ...

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)

    # city와 country가 같은 케이스 찾기
    city_country_pairs = users_.dropna(subset = 'location_state')
    city_country_pairs = city_country_pairs.groupby(['location_city', 'location_country'])['location_state'].nunique()[users_.groupby(['location_city', 'location_country'])['location_state'].nunique() > 1].reset_index().dropna().sort_values('location_state')

    # 각 city-country 쌍에 대해 최빈 state 찾기
    for _, row in city_country_pairs.iterrows():
        city = row['location_city']
        country = row['location_country']
        # 해당 city-country 조합의 모든 데이터 찾기
        condition = (users_['location_city'] == city) & (users_['location_country'] == country) 
        states = users_[condition]['location_state'].dropna()
        # 최빈 state 찾기
        if not states.empty:
            most_common_state = Counter(states).most_common(1)[0][0]
            users_.loc[condition, 'location_state'] = most_common_state # 최빈 state로 업데이트

    # 2. city와 state가 같은 케이스 찾기
    state_city_pairs = users_.dropna(subset = 'location_country')
    state_city_pairs = state_city_pairs.groupby(['location_city', 'location_state'])['location_country'].nunique()[users_.groupby(['location_city', 'location_state'])['location_country'].nunique() > 1].reset_index().dropna().sort_values('location_country')           
    
    # 각 city-country 쌍에 대해 최빈 state 찾기
    for _, row in state_city_pairs.iterrows():
        city = row['location_city']
        state = row['location_state']
        # 해당 city-country 조합의 모든 데이터 찾기
        condition = (users_['location_city'] == city) & (users_['location_state'] == state)
        country = users_[condition]['location_country'].dropna()
        # 최빈 state 찾기
        if not country.empty:
            most_common_country = Counter(country).most_common(1)[0][0]
            users_.loc[condition, 'location_country'] = most_common_country # 최빈 state로 업데이트
   
    # 나머지 결측치는 최빈값으로 대체
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']): # state는 있고 country는 없는 경우 
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']): # city는 있고 state는 없는 경우
            if not pd.isna(row['location_country']): # country는 있는 경우 
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else: # country는 없는 경우
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state
    
        
    users_ = users_.drop(['location'], axis=1)
    books_ = books_.drop(['isbn_code', 'category_clustered', 'category_clustered_clustered'], axis=1)
    return users_, books_

def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_context_data(users, books)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'}) if args.model == 'NCF' \
                   else user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')[sparse_cols]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
    
    
    field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']

    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }

    return data


def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def context_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
