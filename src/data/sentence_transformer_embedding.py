import pandas as pd
import numpy as np
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer

# 불용어 리스트
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 
    'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 
    'shouldn', 'wasn', 'weren', 'won', 'wouldn'
])

# 전처리 함수
def preprocess_text(text: str) -> str:
    """
    텍스트를 소문자로 변환, 불용어와 구두점 제거 후 반환
    
    Args:
        text (str): 전처리할 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    text = text.lower()  # 소문자화
    text = text.translate(str.maketrans('', '', string.punctuation))  # 구두점 제거
    words = text.split()  # 단어로 분할
    words = [word for word in words if word not in stop_words]  # 불용어 제거
    
    return ' '.join(words)  # 결과를 문자열로 결합


# 데이터 프레임 전처리 함수
def books_preprocessing(books: pd.DataFrame) -> pd.DataFrame:
    """
    'book_title', 'category', 'summary' 열의 텍스트 데이터 전처리
    
    Args:
        books (pd.DataFrame): 책 정보를 포함한 데이터프레임
        
    Returns:
        pd.DataFrame: 전처리가 완료된 데이터프레임
    """
    # NaN 값을 빈 문자열로 대체하고 문자열로 변환 후 전처리 적용
    for column in ['book_title', 'category', 'summary']:
        books[column] = books[column].fillna('').astype(str).apply(preprocess_text)
    return books


def main(books: pd.DataFrame, target_dim: int, output_path: str) -> torch.Tensor:
    """
    책 정보에서 문장 임베딩을 생성한 후 MLP 모델로 차원을 축소하여 `.npy` 파일로 저장
    
    Args:
        books (pd.DataFrame): 책 정보를 포함한 데이터프레임
        target_dim (int): 축소하고자 하는 임베딩 차원
        output_path (str): 결과를 저장할 `.npy` 파일 경로
        
    Returns:
        torch.Tensor: 축소된 임베딩 결과
    """
    # 제목, 카테고리, 요약을 하나의 문자열로 합쳐 임베딩 입력 준비
    docs = (books['book_title'] + " " + books['category'] + " " + books['summary']).tolist()
    
    # SentenceTransformer 모델을 사용해 문장 임베딩 생성
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_model.encode(docs)
    
    # 차원 축소를 위한 MLP 모델 정의
    class DimensionReducer(nn.Module):
        """
        MLP 모델을 통해 차원을 축소하는 클래스
        
        Attributes:
            fc1 (nn.Linear): 첫 번째 완전 연결층
            relu (nn.ReLU): ReLU 활성화 함수
            fc2 (nn.Linear): 두 번째 완전 연결층
        """
        def __init__(self):
            super(DimensionReducer, self).__init__()
            self.fc1 = nn.Linear(384, 128)  
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, target_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            입력 텐서의 차원을 축소
            
            Args:
                x (torch.Tensor): 원본 임베딩 텐서
                
            Returns:
                torch.Tensor: 차원이 축소된 텐서
            """
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    # 학습 설정
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    reducer_model = DimensionReducer()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reducer_model.parameters(), lr=0.001)
    target_embeddings = embeddings_tensor[:, :target_dim]  # 목표 차원의 임베딩 설정
    
    # 모델 학습
    num_epochs = 10
    for epoch in range(num_epochs):
        reducer_model.train()
        outputs = reducer_model(embeddings_tensor)
        loss = criterion(outputs, target_embeddings)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 차원 축소된 임베딩 반환 및 저장
    reducer_model.eval()
    with torch.no_grad():
        reduced_embeddings = reducer_model(embeddings_tensor)
    
    # 넘파이 배열로 변환하여 npy 파일로 저장
    reduced_embeddings_np = reduced_embeddings.numpy()
    np.save(output_path, reduced_embeddings_np)
    print(f"Reduced embeddings saved to {output_path}")
    
    return reduced_embeddings
