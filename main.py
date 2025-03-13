import pandas as pd
pd.set_option('future.no_silent_downcasting', True) #чёт жаловалось на это
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_train = pd.read_csv("C:/Users/YAMI/Desktop/Titanic/train.csv")

df_train.head()  # просмотр первых строк
print(df_train.head())
nan_matrix = df_train.isnull()
print(nan_matrix.sum())

#df_without_name = df_train.drop('Name', axis='columns')
df_train.drop('Name', axis='columns', inplace= True)
#Чё делать с именами - хз, с ними всё равно ничего интересного сделать не получится,просто весь столбец имён дропнуть

print(df_train)
df_train.fillna({
    'Age': df_train['Age'].median(),
    'RoomService': df_train['RoomService'].median(),
    'FoodCourt': df_train['FoodCourt'].median(),
    'VRDeck': df_train['VRDeck'].median(),
    'Spa': df_train['Spa'].median(),
    'ShoppingMall': df_train['ShoppingMall'].median(),
    'CryoSleep': df_train['CryoSleep'].mode()[0],
    'Cabin': df_train['Cabin'].mode()[0],
    'Destination': df_train['Destination'].mode()[0],
    'VIP': df_train['VIP'].mode()[0],
    'HomePlanet': df_train['HomePlanet'].mode()[0]
}, inplace=True)




numerical_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])



df_train = pd.get_dummies(df_train, columns=['HomePlanet'], drop_first=True)
#Надо ли делать категоризацию по каждому полю?

df_train.to_csv("C:/Users/YAMI/Desktop/Titanic/processedTrain.csv", index=False)