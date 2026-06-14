import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

df=pd.read_csv("match_data.csv")

print("Raw Data:")
print (df)
print("\nshape:",df.shape)

#data cleaning
df= df[df["Winner"] != "No Result"] #removed no result row
df=df.dropna()

le_team1=LabelEncoder()
le_team2=LabelEncoder()
le_venue=LabelEncoder()
le_winner=LabelEncoder()

df["Team1_enc"]=le_team1.fit_transform(df["Team1"])
df["Team2_enc"]=le_team2.fit_transform(df["Team2"])
df["Venue_enc"]=le_venue.fit_transform(df["Venue"])
df["Winner_enc"]=le_winner.fit_transform(df["Winner"])

print("\nEncoded data:")
print(df[["Team1","Team1_enc","Team2","Team2_enc","Winner","Winner_enc"]])

x=df[["Team1_enc","Team2_enc","Venue_enc"]]
y=df["Winner_enc"]

print("\nfeaturs (x):")
print(x)
print("\nTarget (y):")
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("\ntraining Samples:",len(x_train))
print("Testing Samples:",len(x_test))

model=LogisticRegression()
model.fit(x_train,y_train)

print("\nModel trained successfully!!")

y_pred=model.predict(x_test)
print("\nActual Winners :",list(le_winner.inverse_transform(y_test)))
print("predicted Winners:",list(le_winner.inverse_transform(y_pred)))

accuracy=accuracy_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,average="weighted",zero_division=0)
cm=confusion_matrix(y_test,y_pred)

print(f"Accuracy Score:{accuracy*100:.2f}%")
print(f"F1 Score :{f1:.2f}")
print(f"Confusing Matrix:\n{cm}")
