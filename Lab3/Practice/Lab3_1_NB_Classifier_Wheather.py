from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB

weather = ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy','Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy']
temp = ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild', 'Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes', 'Yes','Yes','Yes','Yes','No']


#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print("Weather:" ,weather_encoded)

temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

print("Temp:",temp_encoded)
print("Play:",label)

#Combinig weather and temp into single listof tuples
features=tuple(zip(weather_encoded,temp_encoded))
print("Features:",features)

#Create a Classifier
model=MultinomialNB()
# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)

predicted= model.predict([[0,1]]) # 0:Overcast, 1:Hot
print("Predicted Value:", predicted)

predicted= model.predict([[2,2]]) # 2:Sunny, 2:Mild

print("Predicted Value:", predicted)

