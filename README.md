# sherpa2

`models.py` contains the pytorch model classes. For now, we use the sherpamodel only which essentially is a version of the original sherpa model found in the sherpa repository but in pytorch instead of tensorflow. It might be that the sherpamodel here is a little bit bigger than the original one but it is not vastly different. 

`classification_model.py` handles the hyperparameter optimization (*objective*) and the training of the model (*train_model*) using kfold cross validation (*run_kfold_evaluation*). We first split of a random hold out test set (10%) and retrain the model on all data except this hold out test set after cross validation to get a good estimate of generalization ability (within the dataset). 

The *argumentparser* makes sure we can run the training and testing from `run_all_sources.py`to make it easier to switch between processes and models and run the different parts of the dataset back to back. 

Lastly, `classification_model.py` runs SHAP as a basis for SHERPA. 

`sherpa.py` is not in the last state for the moment and is not used right now as everything regarding to that is run in Matlab.
