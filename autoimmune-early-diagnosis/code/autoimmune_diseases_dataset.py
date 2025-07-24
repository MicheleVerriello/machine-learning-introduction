import pandas as pd

class AutoimmuneDiseasesDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load(self):
        self.data = pd.read_csv(self.filepath)
        return self.data

    @property
    def X(self):
        if self.data is None:
            self.load()
        X = self.data.drop(columns=['Patient_ID', 'Diagnosis'])
        
        # Encode Gender: Male=0, Female=1
        if 'Gender' in X.columns:
            X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})
        return X



    @property
    def y(self):
        if self.data is None:
            self.load()
        return self.data['Diagnosis']