# :palm_tree: Animal Crossing Villager Popularity Predictor + Recommendation System
I love Animal Crossing. I love data. Therefore, I built this machine learning project to:
* Predict the **popularity** of Animal Crossing: New Horizons villagers based on image and metadata 
  features.
* Recommend **similar villagers** based on your favorites using cosine similarity.

This is currently a work-in-progress.
## Project Structure
```
AnimalCrossing/
├── data/           # Raw, interim, and processed datasets
├── scripts/        # Modules for preprocessing, feature engineering, modeling
├── outputs/        # Generated predictions, plots, and logs
├── requirements.txt
└── README.md
```

## Data Sources
| Source                                                            | Link                                                                                                                     | Description                                                |
|-------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| Data Spreadsheet for Animal Crossing New Horizons | [Google Sheets](https://docs.google.com/spreadsheets/d/13d_LAJPlxMa_DubPTuirkIV4DERBMXbrWQsmSh8ReK4/edit?usp=drive_link) | Rich metadata on all Animal Crossing villagers.            |
| Animal Crossing Villager Popularity Tier List                     | [Animal Crossing Portal](https://www.animalcrossingportal.com/tier-lists/new-horizons/all-villagers/)                    | Crowdsourced popularity data. |

## How to Run
### Environment Setup
``` bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### Run Popularity Prediction
```bash
python scripts/modeling/prediction_model.py
```
Outputs:
* `outputs/test_predictions.csv`: Predicted vs. actual popularity
* `outputs/top_k_coefficients.png`: Top k most influential model features
### Run Recommendation System
```bash
python scripts/modeling/recommender.py
```
Outputs:
* Console-printed recommended villagers most similar to your selected favorites
## GOAL 1: Predicting Villager Popularity
**Type**: Supervised Regression Task  
**Objective**: Predict how popular a given Animal Crossing: New Horizons villager is based on both 
structured metadata and unstructured image features.

We use a Ridge Regression model to balance interpretability and performance, and to allow 
feature weighting between categorical and image-based data.
### Features
#### 1. Categorical Features (One-Hot Encoded)
* `Species`: Cat, Dog, Deer, etc.
* `Gender`: Male / Female
* `Personality`: Lazy, Peppy, etc.
* `Personality Subtype`: Lazy_A, Lazy_B, etc.
* `Hobby`: Music, Nature, etc.
* `Style 1`, `Style 2`: Active, Gorgeous, etc.
* `Color 1`, `Color 2`: Aqua, Grey, etc.

We also include unordered style/color sets using `MultiLabelBinarizer` to capture overlap in 
  aesthetic preferences (for example Gorgeous + Cool = same tag whether order is flipped).
#### 2. Visual Features
Using OpenAI CLIP, we extract 512-dimensional CLIP embeddings from:
* `Icon Image` (game profile headshot)
* `Photo Image` (in-game poster)

We then apply PCA to reduce dimensionality.

#### 3. Textual Features (WIP)
This has not been currently implemented yet. These are good candidates for word2vec or 
sentence-BERT.
* `Catchphrase`
* `Favorite Saying`

### 4. Ignored Misc. Features
The following are features that probably do not have a significant impact on villager popularity.
In the future, they could be implemented.
* `Favorite Song`
* `Default Umbrella`
* `Wallpaper`
* `Flooring`
* `Birthday`
* `Furniture List`
* `Furniture Name List`
* `Version Added`

## GOAL 2: Villager Recommender System
**Type**: Unsupervised Similarity Matching  
**Objective**: Given a small list of favorite villagers, recommend others that share similar 
characteristics or aesthetics.

### Methodology
* Select a few favorite villagers.
* Construct the feature vector for each.
* Compute the mean feature vector across your selected favorites.
* Calculate cosine similarity between this average and all other villagers.
* Recommend the top $k$ villagers with highest similarity scores.
