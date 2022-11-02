Directory xaverius777/diss/ccrater contains the following:
+ CCRater.py - Given a noisy-enhanced speech audio pair, produces a CCR score. File includes the model itself plus the code infrastructure that was used to train and evaluate the model. 
+ session_log_103.txt - Contains the MSE loss on the dev set for all the epochs during the training
+ training_log - Apart from session_log_103.txt information, it also includes the model's performance on the training set's batches
+ CC_Rater_69.pt - Trained weights for the model that produced the best results on the dev set (MSE loss = 0.096)
+ variables - Needed for running the DNSMOS block inside of CC-Rater. These variables were obtained when the PyTorch code was generated from Microsoft's ONNX file for DNSMOS, sig_bak_ovr.onnx
All information about DNSMOS can be found in this link: https://github.com/microsoft/DNS-Challenge
Package used for converting ONNX to PyTorch: https://github.com/fumihwh/onnx-pytorch
