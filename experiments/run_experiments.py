import pandas as pd 
import numpy as np
import pickle, warnings
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mutual_info_score, auc
# supress all the warnings
import warnings
warnings.filterwarnings("ignore")

# Load the data
targets = pd.read_csv("multi-media-interaction/experiments/targets/restaurants.csv")

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data 

def extract_zero_shot_embeddings(data, pretrained_model):   
    restaurants = []
    features = []
    if pretrained_model == 'FLAVA':
        for restaurant in data:
            feature = restaurant['embedding']
            feature = feature.mean(axis=0)
            restaurants.append(restaurant['restaurant'])
            features.append(feature)
    elif pretrained_model == 'CLIP':
        for restaurant in data:
            text = restaurant['text']
            image = restaurant['image']
            text = text.mean(axis=0)
            image = image.mean(axis=0)
            feature = np.concatenate([text, image])
            restaurants.append(restaurant['restaurant'])
            features.append(feature)
    else:
        raise ValueError('Pretrained model not supported.')
    return features, restaurants


def fit(data, target, weighted, pretrained_model, model_type='logistic'):

    # average embeddings, textual and visual 
    features, restaurants = extract_zero_shot_embeddings(data, pretrained_model)    

    # train test split
    df = pd.DataFrame(features, index=restaurants).reset_index()
    df = df.merge(targets, left_on=('index'), right_on=('location_name'), how='inner').dropna(subset=[target])
    if target == 'price':
        df['price'] = df['price'].astype(int) - 1
    X = df.drop(columns=['index', 'location_name', 'is_closed', 'review_count', 'rating', 'price', 'is_closed'])
    y = df[target].astype(int)

    # weights 
    weights = y.value_counts()
    if weighted:
        if weights.shape[0] == 2:
            weightss = weights[0] / weights[1]
            class_weights = {0:1, 1:weightss}
        else:
            m = len(y)
            class_weights = {int(price): m / (len(weights) * weights.loc[price]) for price in weights.index}
    else:
        class_weights = None
    
    if len(weights) == 2:
        scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc', 'mutual_info_score']
    else:
        scoring=['accuracy', 'f1_micro', 'precision_micro', 'recall_micro', 'roc_auc_ovr', 'mutual_info_score', 'neg_mean_squared_error', 'neg_mean_absolute_error']


    # fit the model
    if model_type == 'logistic':
        model = LogisticRegression(random_state=46, class_weight=class_weights, penalty='l1', solver='saga')
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=46, class_weight=class_weights)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=46, class_weight=class_weights)
    elif model_type == 'boosting':
        model = XGBClassifier(random_state=46, class_weight=class_weights, learning_rate=0.1)
    elif model_type == 'regression':
        model = LinearRegression()
    else:
        raise ValueError('Model type not supported.')
    
    model_pipeline = ('classifier', model) if model_type != 'regression' else ('regressor', model)
    scoring = scoring if model_type != 'regression' else ['neg_mean_squared_error', 'neg_mean_absolute_error']
    pipeline = Pipeline([
        ('pca', PCA(n_components=.9, random_state=46)), 
        model_pipeline])
    
    cv_results = cross_validate(pipeline, X, y, cv=5, scoring=scoring)
    print(f'Sample Size: {len(X)}')
    for key, value in cv_results.items():
        print(f"{key}:", value.mean())


if __name__ == '__main__':
    embeddings = 'fully_trained' # 'zero_shot', 'fine_tuned', 'fully_trained'
    sample = 10
    pretrained_model = "CLIP"
    target = 'is_closed'
    model_type = 'logistic'
    weighted = True
    if embeddings == 'zero_shot':
        path = "multi-media-interaction/experiments/CLIP/zero_shot_embeddings/zero_shot_embeddings.pkl" # Zero-shot Embeddings
    elif embeddings == 'fine_tuned':
        path = "multi-media-interaction/experiments/CLIP/fine_tuned_embeddings/fine_tuned_embeddings.pkl" # Fine-tuned Embeddings 
    elif embeddings == 'fully_trained':
        path = "multi-media-interaction/experiments/CLIP/paper_replication/fully_trained_embeddings.pkl" # Fully trained Embeddings 
    else:
        raise ValueError('Embeddings not supported.')
    data = load_data(path)
    print('Pretrained Model:', pretrained_model, 'Embeddings:', embeddings,'# Samples:', sample, 'Model:', model_type, 'Weighted:', weighted, 'Target:', target)
    fit(data, target, weighted, pretrained_model, model_type=model_type)
