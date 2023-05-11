The first big improve is using StandardScaler, loss reduce from 0.3 to 0.05. MinMaxScaler works same.
Try use train set to fit scaler. Not much difference with use pretrain set to fit.
Try use smaller batch size for train set, but it overfitted with loss as 0.02 and lead to bad score.
