# -*- coding: utf-8 -*-
# W6 Titanic Preprocessing Template
# 僅可修改 TODO 區塊，其餘部分請勿更動
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
 
# 任務 1：載入資料
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)
 
 
# 任務 2：處理缺失值
def handle_missing(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.isnull().sum()
    return df
 
 
# 任務 3：移除異常值
def remove_outliers(df):
    prev_len = 0
    while len(df) != prev_len:  # 當資料筆數還在變動時繼續
        prev_len = len(df)
        fare_mean = df['Fare'].mean()
        fare_std = df['Fare'].std()
        threshold = fare_mean + 3 * fare_std
        df = df[df['Fare'] <= threshold]  # 保留正常範圍
    return df
 
 
# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df_encoded = pd.concat([df.drop(['Sex', 'Embarked'], axis=1),sex_dummies, embarked_dummies], axis=1)
    return df_encoded
 
 
# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    df_scaled = df
    return df_scaled
 
 
# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
 
 
# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, encoding='utf-8-sig', index=False)
 
 
# 主程式流程（請勿修改）
if __name__ == "__main__":
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"
 
    df, missing_count = load_data(input_path)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(df, output_path)
 
    print("Titanic 資料前處理完成")