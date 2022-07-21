import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataMixer:
    def __init__(self, non_iid_path):
        '''
            Creates a new shared data partition that can be processed by the participants.
            Parameters:
                -non_iid_path: path to noniid_50 participants' data
        '''
        self.participants_data = []
        for partip in range(10):
            self.participants_data.append(pd.read_csv(non_iid_path+"participant%d.csv"%partip))

    def create_new_shared_data(self, samples_to_share):
        '''
            Creates a new shared data partition that can be processed by the participants.
            Parameters:
                - samples_to_share: how many samples has to be shared from each participants' data per digit.
            Returns:
                - a Pandas.DataFrame containing the new shared data
        '''
        shared_df = None
        for partip in range(10):
            participant_df = self.participants_data[partip]
            for digit in range(10):
                digit_df = participant_df[participant_df["label"] == digit].sample(n=samples_to_share).reset_index(drop=True)
                shared_df = pd.concat([shared_df, digit_df])
        return shared_df.reset_index(drop=True)
    
    