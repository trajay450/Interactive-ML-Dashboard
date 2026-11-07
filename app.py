"""Streamlit dashboard for an independently trained public digit classifier."""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dashboard.model import train_digits

st.set_page_config(page_title='ML Evaluation Dashboard',layout='wide')
st.title('Interactive ML Evaluation Dashboard')
st.caption('Public scikit-learn digits dataset • Logistic regression baseline • Educational demonstration')

seed=st.sidebar.number_input('Random seed',min_value=0,max_value=100000,value=42)
@st.cache_resource(show_spinner=False)
def cached_train(seed): return train_digits(seed)
results=cached_train(seed)
left,right=st.columns(2)
left.metric('Held-out accuracy',f"{results['metrics']['accuracy']:.2%}")
right.metric('Macro-F1',f"{results['metrics']['macro_f1']:.3f}")
st.subheader('Confusion matrix')
fig,ax=plt.subplots(figsize=(7,5))
img=ax.imshow(results['confusion_matrix'],cmap='Blues')
ax.set(xlabel='Predicted digit',ylabel='True digit')
ax.set_xticks(np.arange(10));ax.set_yticks(np.arange(10))
fig.colorbar(img,ax=ax)
st.pyplot(fig);plt.close(fig)
st.subheader('Sample prediction explorer')
idx=st.slider('Test example',0,len(results['truth'])-1,0)
col1,col2=st.columns([1,2])
fig,ax=plt.subplots(figsize=(3,3))
ax.imshow(results['images'][idx],cmap='gray');ax.axis('off')
col1.pyplot(fig);plt.close(fig)
col2.write(f"True digit: **{results['truth'][idx]}**")
col2.write(f"Predicted digit: **{results['pred'][idx]}**")
col2.bar_chart(pd.DataFrame({'Probability':results['probabilities'][idx]},index=np.arange(10)))
st.subheader('Per-class metrics')
report=pd.DataFrame(results['report']).T
st.dataframe(report.loc[[str(i) for i in range(10)],['precision','recall','f1-score','support']])
st.caption('This dashboard does not diagnose patients and is not connected to unpublished research.')
