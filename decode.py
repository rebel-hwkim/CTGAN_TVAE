from ctgan import TVAE
from ctgan import load_demo
from ctgan.synthesizers.tvae import Decoder
import torch
from ctgan.data_transformer import DataTransformer

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
tvae.transformer = DataTransformer()
tvae.transformer.fit(real_data, discrete_columns)
tvae.decoder = Decoder(tvae.embedding_dim, tvae.decompress_dims, tvae.transformer.output_dimensions).to(tvae._device)
tvae.decoder.load_state_dict(torch.load('decoder_128.pt'))

synthetic_data = tvae.sample(1024)
print(synthetic_data)