from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 
from sklearn.linear_model import BayesianRidge 
  
# Loading dataset 
dataset = load_boston() 
X, y = dataset.data, dataset.target 
  
# Splitting dataset into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42) 
  
# Creating and training model 
model = BayesianRidge() 
model.fit(X_train, y_train) 
  
# Model making a prediction on test data 
prediction = model.predict(X_test) 
  
# Evaluation of r2 score of the model against the test set 
print("r2 Score Of Test Set : "+(str)(r2_score(y_test, prediction)))