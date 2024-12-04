# 📚 Book Recommendation

책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점을 활용하여 **사용자가 주어진 책에 대해 얼
마나 평점을 부여**할지에 대해 예측합니다.
해당 경진대회는 소비자들의 책 구매 결정에 대한 도움을 주기 위한 **개인화된 상품 추천 대회**입니다.

자세한 프로젝트 내용은 [Wrap-up Report](https://github.com/boostcampaitech7/level2-bookratingprediction-recsys-02/blob/main/Wrap_Up_Report_Book_Recommendation.pdf) 를 통해 확인해주세요.

## 💡Team
#### 팀 구성
| [강현구](https://github.com/ardkyer) | [서동준](https://github.com/seoo2001) | [이도걸](https://github.com/doffice0827) | [이수미](https://github.com/SooMiiii) | [최윤혜](https://github.com/yunhyechoi) | 
| --- | --- | --- | --- | --- | 
| <img src="https://github.com/user-attachments/assets/e00fe2c2-20d6-497e-8d15-32368381f544" width="150" height="150"/> | <img src="https://github.com/user-attachments/assets/674a4608-a446-429f-957d-1bebeb48834f" width="150" height="150"/> | <img src="https://github.com/user-attachments/assets/1bdbd568-716a-40b7-937e-cbc5b1e063b8" width="150" height="150"/> | <img src="https://github.com/user-attachments/assets/c8fc634a-e41e-4b03-8779-a18235caa894" width="150" height="150"/> | <img src="https://github.com/user-attachments/assets/7a0a32bc-d22c-47a1-a6c7-2ea35aa7b912" width="150" height="150"/> 

#### 역할 분담
| 이름 | 역할 |
| --- | --- |
| 강현구 | EDA, DeepFM, ResNet_DeepFM |
| 서동준 | TextFM, TextDeepFM, LGBM |
| 이도걸 | WDN, DCN, NCF, Boosting |
| 이수미 | Add features, FM, FFM |
| 최윤혜 | Data Preprocessing, ImageFM, Image DeepFM, CatBoost |

## 📂 Architecture
```
.
📦 Project Root
┣ 📜 main.py
┣ 📂 src
┃ ┣ 📂 data
┃ ┃ ┣ 📊 basic_data.py
┃ ┃ ┣ 📊 context_data.py
┃ ┃ ┣ 🖼️ image_data.py
┃ ┃ ┗ 📝 text_data.py
┃ ┣ 📂 ensembles
┃ ┃ ┗ 🤝 ensembles.py
┃ ┣ 📂 loss
┃ ┃ ┗ 📉 loss.py
┃ ┣ 📂 models
┃ ┃ ┣ 🛠️ _helpers.py
┃ ┃ ┣ 🔮 DCN.py
┃ ┃ ┣ 🔮 DeepFM.py
┃ ┃ ┣ 🔮 FFM.py
┃ ┃ ┣ 🖼️ FM_Image.py
┃ ┃ ┣ 📝 FM_Text.py
┃ ┃ ┣ 🔮 FM.py
┃ ┃ ┣ 🔮 NCF.py
┃ ┃ ┗ 🔮 WDN.py
┃ ┣ 📂 train
┃ ┃ ┗ 🎯 trainer.py
┃ ┗ 🛠️ utils.py
┣ 📜 ensemble.py
┣ 📜 run_baseline.sh
┣ 📓 eda
┃ ┗ 📊 eda.ipynb
┗ 📂 config
┣ ⚙️ config_baseline
┗ ⚙️ sweep_example
```

## ⚒️ Development Environment

- 서버 스펙 : AI Stage GPU (Tesla V100)
- 협업 툴 : Github / Zoom / Slack / Notion / Google Drive
- 기술 스택 : Python / Scikit-Learn / Scikit-Optimize / Pandas / Numpy / PyTorch

