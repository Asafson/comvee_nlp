import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from constants import classes_str_to_num

le = LabelEncoder()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
label_mapper = {'Non-Manager': None, 'Manager': 'Manager-Level', 'Director': 'Director-Level'}
column_mapping_for_model = {"text": "text", "label_num": "label"}


def split_rename():
    df = pd.read_csv('/Users/asaflev/Desktop/Centrical Intail List.csv')
    df = df[['Title', 'Management Level']].rename(columns={'Title': 'text', 'Management Level': 'label_str'})
    df.dropna(subset=['text'], inplace=True)
    management_level_counts = df['label_str'].value_counts(dropna=False)
    df['label_str'] = df['label_str'].apply(lambda x: label_mapper.get(x) if x in label_mapper else x)
    df['label_str'] = df['label_str'].fillna(value='none')
    df['label_num'] = df['label_str'].map(classes_str_to_num)
    df_majority, df_sample = train_test_split(df, test_size=.02, stratify=df['label_num'])
    train_df, test_df = train_test_split(df_sample, test_size=.3, stratify=df_sample['label_num'])
    return train_df, test_df


def get_model_conf(model, train_ds, val_ds):
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        batch_size=16,
        num_iterations=20,
        num_epochs=1,
        column_mapping=column_mapping_for_model
    )
    return trainer


def get_metrics(df_val):
    sns.heatmap(confusion_matrix(df_val['label_num'], df_val['pred']),
                annot=True,
                square=True)
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.show()


def save_model(set_fit_model):
    #Here you can change the path as you with - same path needed to be loaded in app
    joblib.dump(set_fit_model, 'model.joblib')


def main():
    df_train, df_val = split_rename()
    train_ds = Dataset.from_pandas(df_train)
    val_ds = Dataset.from_pandas(df_val)
    logger.info('Made data sets for model')
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")
    logger.info('Start training setfit model')
    set_fit_model = get_model_conf(model, train_ds, val_ds)
    set_fit_model.train()
    df_val['pred'] = set_fit_model.model.predict(df_val.reset_index()['text'])
    get_metrics(df_val)
    save_model(set_fit_model)
    return set_fit_model


if __name__ == '__main__':
    set_fit_model = main()
