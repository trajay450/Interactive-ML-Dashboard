"""Train a digits classifier with reproducible stratified hold-out."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_digits(seed=42):
    ds=load_digits()
    X_train,X_test,y_train,y_test=train_test_split(ds.data,ds.target,test_size=0.25,
                                                    stratify=ds.target,random_state=seed)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,random_state=seed))
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    metrics={'accuracy':float(accuracy_score(y_test,pred)),
             'macro_f1':float(f1_score(y_test,pred,average='macro'))}
    return {'model':model,'images':X_test.reshape(-1,8,8),'truth':y_test,'pred':pred,
            'probabilities':model.predict_proba(X_test),
            'confusion_matrix':confusion_matrix(y_test,pred,labels=np.arange(10)),
            'report':classification_report(y_test,pred,output_dict=True,zero_division=0),
            'metrics':metrics}
