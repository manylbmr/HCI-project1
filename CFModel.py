# A simple implementation of matrix factorization for collaborative filtering expressed as a Keras functional model

import numpy as np
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten
from tensorflow.keras.models import Model

class CFModel(Model):
    def __init__(self, n_users, m_items, k_factors, **kwargs):
        super(CFModel, self).__init__(**kwargs)

        # Define inputs
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')

        # Define user embedding
        user_embedding = Embedding(input_dim=n_users, output_dim=k_factors, name='user_embedding')(user_input)
        user_vec = Reshape((k_factors,), name='user_vector')(user_embedding)

        # Define item embedding
        item_embedding = Embedding(input_dim=m_items, output_dim=k_factors, name='item_embedding')(item_input)
        item_vec = Reshape((k_factors,), name='item_vector')(item_embedding)

        # Dot product of user and item vectors
        dot_product = Dot(axes=1, name='dot_product')([user_vec, item_vec])

        # Build model
        self.model = Model(inputs=[user_input, item_input], outputs=dot_product)

    def call(self, inputs):
        return self.model(inputs)

    def rate(self, user_id, item_id):
        user_id = np.array([user_id])
        item_id = np.array([item_id])
        return self.model.predict([user_id, item_id])[0][0]
