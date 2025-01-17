from ctgan import TVAE
from ctgan import load_demo
import torch

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

batch_size = 512
embedding_dim = 128
tvae = TVAE(epochs=10, batch_size=batch_size, embedding_dim=embedding_dim, cuda=True)
tvae.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = tvae.sample(100)
print(synthetic_data)
