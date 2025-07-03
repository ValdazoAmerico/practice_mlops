import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

iris_data=pd.read_csv('data/test.csv')
print("len,",len(iris_data))



X=iris_data.drop(columns='Species',axis=1)
Y=iris_data['Species']
models=[LogisticRegression(max_iter=100),SVC(),KNeighborsClassifier(),RandomForestClassifier()]
from sklearn.preprocessing import StandardScaler
#scaler=StandardScaler()
#scaler.fit(X)
StandardScaler()
#X=scaler.transform(X)


train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=0)
print("train x",train_x)
print("len x",len(train_x))
print("len y",len(train_y))
print("len test x",len(test_x))
print("len test y",len(test_y)) 

#model=SVC(kernel='rbf',C=5)
model=models[0]
model.fit(train_x,train_y)

train_x_prediction=model.predict(train_x)
print("pred",train_x_prediction)
from sklearn.metrics import accuracy_score,confusion_matrix
train_x_accuracy=accuracy_score(train_y,train_x_prediction)
print("accuracy train",train_x_accuracy)

test_x_prediction=model.predict(test_x)
test_x_accuracy=accuracy_score(test_y,test_x_prediction)
print("accuracy test",test_x_accuracy)

cf_matrix=confusion_matrix(test_y,test_x_prediction)
print("matrix",cf_matrix)
joblib.dump(model, "models/candidato_model.pkl")