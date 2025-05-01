#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#a function to load and soft the event data by Timestamp
def data_load(data): 
  df = pd.read_csv(data, parse_dates = ['ts_event'])
  return df.sort_values('ts_event').reset_index(drop=True)

#a function to compute the OFI at level m
def compute_ofi(df, m): 
  #get the bid and ask data
  bp = f'bid_px_{m:02d}' 
  ap = f'ask_px_{m:02d}'
  bs = f'bid_sz_{m:02d}' 
  ask = f'ask_sz_{m:02d}'
  #get the previous bid and ask data
  bp_prev = df[bp].shift(1) 
  ap_prev = df[ap].shift(1)
  bs_prev = df[bs].shift(1)
  as_prev = df[ask].shift(1)
  #compute the bid order flow
  ofb = np.where(df[bp] > bp_prev, df[bs], 
      np.where(df[bp] < bp_prev,-df[bs],df[bs] - bs_prev)
  )
  #comoute the ask order flow
  ofa = np.where(df[ap] > ap_prev,-df[ask],
      np.where(df[ap] < ap_prev,df[ask],df[ask] - as_prev
      )
  )
  #compute the ofi
  ofi = ofb - ofa
  return pd.Series(ofi, index=df.index).fillna(0)

def best_level_ofi(df):
  df = df.copy()
  df['OFI_best'] = compute_ofi(df,0)
  return df[['ts_event', 'OFI_best']]

def multi_level_ofi(df, M = 10):
    df = df.copy()
    #create cols for all 10 levels
    ofi_cols = []
    for m in range(M):
        col = f'OFI_{m}'
        #compute ofi at each level
        df[col] = compute_ofi(df, m).fillna(0)
        ofi_cols.append(col)
    df['OFI_multi'] = df[ofi_cols].sum(axis=1)
    return df[['ts_event','OFI_multi'] + ofi_cols]

def integrated_ofi(df, M = 10):
    #get multi level ofi data
    multi = multi_level_ofi(df, M)
    ofi_matrix = multi[[f'OFI_{m}' for m in range(M)]].values
    #fit the Principal Component Analysis
    pca = PCA(n_components=1)
    pca.fit(ofi_matrix)
    pc1 = pca.components_[0]
    #normalize the weights
    w = pc1 / np.abs(pc1).sum()
    integrated = ofi_matrix.dot(w)
    df_out = pd.DataFrame({'ts_event': multi['ts_event'], 'OFI_int': integrated})
    return df_out

def cross_asset_ofi(df_A, df_B):
  #get multi_level ofi data from A and B
  ofi_A = multi_level_ofi(df_A).rename(columns={'OFI_multi':'OFI_A'})
  ofi_B = multi_level_ofi(df_B).rename(columns={'OFI_multi':'OFI_B'})
  #connect the ofi data from A and B
  merged = pd.merge_asof(ofi_A.sort_values('ts_event'), ofi_B.sort_values('ts_event'), on='ts_event', direction='backward')
  return merged[['ts_event','OFI_B']].rename(columns={'OFI_B':'OFI_cross_for_A'})
