import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV



st.set_page_config("AutoScout24 Price Prediction App",page_icon="https://www.autoscout24.com/cms-content-assets/1tkbXrmTEPPaTFel6UxtLr-c0eb4849caa00accfa44b32e8da0a2ff-AutoScout24_primary_solid.png") # page title


img = Image.open("image.png")  # image 

st.image(img, caption="AutoScout24")


st.write("<h1 style='font-family:Courier; background-color:yellow; color:black; font-size: 38px;'>Predict AutoScout24 Car Prices</h1>",unsafe_allow_html=True) #project title
st.markdown("""---""")

# social_acc = ["About","LinkedIn","Github"]
# social_acc_nav = st.sidebar.selectbox("About", social_acc)
# if social_acc_nav == "About":
#     st.sidebar.markdown("<h2 style='text-align: center;'> Auto Scout 24 </h2> ",unsafe_allow_html=True)
#     st.sidebar.markdown("""---""")
#     st.sidebar.markdown("""
#     • Used and New Cars \n
#     • Motorbikes \n
#     • Trucks """)
#st.sidebar.markdown("[ Visit Website](https://www.autoscout24.com/?genlnk=navi&genlnkorigin=tr-all-all-home)")



df = pd.read_csv("new_file.csv")

choice1 = st.selectbox("Select the Model:",("Audi A1","Audi A3","Opel Astra","Opel Corsa","Opel Insignia","Renault Clio","Renault Duster","Renault Espace"))
st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">{"You have selected :"}{choice1}  </h1>', unsafe_allow_html=True)
st.write("\n")
choice2= st.number_input("Enter the Age:",min_value=0)
st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">{"You have selected :"}{choice2}  </h1>', unsafe_allow_html=True)

choice3=st.selectbox("Select the Gearing Type:",("Automatic","Manual"))
st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">{"You have selected :"}{choice3}  </h1>', unsafe_allow_html=True)

choice4=st.number_input("Enter Hp_kW:",min_value=40,max_value=240)
st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">{"You have selected :"}{choice4}  </h1>', unsafe_allow_html=True)

choice5=st.number_input("Enter KM:",min_value=0)
st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">{"You have selected :"}{choice5}  </h1>', unsafe_allow_html=True)

import pickle
alpha_space = np.linspace(0.01, 100, 100)
X = df.drop(columns = ["price"])
y= df.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
final_scaler = MinMaxScaler()
final_scaler.fit(X) 
X_scaled = final_scaler.transform(X)
filename = 'my_model'
model = pickle.load(open(filename, 'rb'))



my_dict = {
    "hp_kW": choice4,
    "age": choice2,
    "km": choice5,
    "make_model": choice1,
    "Gearing_Type": choice3
}

my_dict = pd.DataFrame.from_dict([my_dict])
my_dict = pd.get_dummies(my_dict)
my_dict = my_dict.reindex(columns = X.columns, fill_value=0) 
my_dict = final_scaler.transform(my_dict) 

pr = model.predict(my_dict)
price= round(pr[0],3)


    
if st.button("Predict") and price > 0 :
    pr = model.predict(my_dict)
    price= round(pr[0],3)
    st.balloons()       
    st.write(f'<h1 style="font-family:Courier; background-color:yellow;opacity: 0.9;text-align: center; color:black;;font-size:16px;">The price of the Car : ${price} </h1>',unsafe_allow_html=True)
          
if price < 0 :
    st.write(f'<h1 style="font-family:Courier; background-color:black;opacity: 0.9;text-align: center; color:red;;font-size:16px;">!!! Wrong choice made !!! </h1>',unsafe_allow_html=True)
   
