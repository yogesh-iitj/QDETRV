import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    df = pd.read_csv('/data1/saswats/baseline/os2d/baselines/CoAE/data_new_open/grozi/classes/grozi_listed.csv')

    n_box = []
    for i in tqdm(range(len(df))):
        for box in df['boxes'][i][2:-2].split(","):
            box = box.replace("\"", "").replace("'", "").replace(" ", "").split('_')
            b = [float(val) for val in box]
            # print(b)
            n_box.append(b)
        n_box = np.array(n_box)
        df['boxes'][i] = n_box
        n_box = []
    
    df.to_csv('/data1/yogesh/one-shot-det-vid/dataset/final_split/new_data.csv', index=False)

if __name__ == '__main__':
    main()

