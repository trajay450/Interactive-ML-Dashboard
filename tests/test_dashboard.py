import numpy as np
from dashboard.model import train_digits

def test_shapes_metrics():
    r=train_digits()
    assert r['confusion_matrix'].shape==(10,10)
    assert len(r['images'])==len(r['truth'])==len(r['pred'])
    assert np.allclose(r['probabilities'].sum(axis=1),1)
    assert 0<=r['metrics']['accuracy']<=1

def test_reproducible():
    assert train_digits(42)['metrics']==train_digits(42)['metrics']
