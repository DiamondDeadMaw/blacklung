# blacklung
 
Final.py contains the XGBoost model \
FinalTrans.py contains Transformer v1 \
preprocess processes the full dataset (not included) \
train.py and eval.py allow the user to switch between LSTM
and Transformer v2, which are implemented in transformer_model.py
and lstm_model.py \

To use Final and FinalTrans you will have to edit compressor.py
to decompress the Final10Stations.csv.xz file by changing the respective flag.
