import pickle
import os
import numpy as np
from lightfm import LightFM
import config

class HybridLightFM:
    def __init__(self, 
                 no_components=config.NO_COMPONENTS, 
                 learning_rate=config.LEARNING_RATE, 
                 loss=config.LOSS):
        
        self.model = LightFM(no_components=no_components,
                             learning_rate=learning_rate,
                             loss=loss)
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.loss = loss
        
    def fit(self, interactions, item_features=None, epochs=1, verbose=True, num_threads=config.NUM_THREADS):
        self.model.fit(interactions, 
                       item_features=item_features, 
                       epochs=epochs, 
                       verbose=verbose, 
                       num_threads=num_threads)
                       
    def fit_partial(self, interactions, item_features=None, epochs=1, verbose=True, num_threads=config.NUM_THREADS):
        self.model.fit_partial(interactions, 
                               item_features=item_features, 
                               epochs=epochs, 
                               verbose=verbose, 
                               num_threads=num_threads)

    def predict(self, user_ids, item_ids, item_features=None, num_threads=config.NUM_THREADS):
        return self.model.predict(user_ids, item_ids, item_features=item_features, num_threads=num_threads)

    def save(self, filepath):
        print(f"Saving model to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, filepath):
        print(f"Loading model from {filepath}")
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
            
    def get_repr(self):
        return f"HybridLightFM(components={self.no_components}, lr={self.learning_rate}, loss={self.loss})"
