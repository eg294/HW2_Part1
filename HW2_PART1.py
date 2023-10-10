import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

#===============================Question 1===============================
# Replace 'imports-85.csv' with the actual file path if it's not in the current directory
dataset_url = "/home/eg294/Documents/CS370/imports-85.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(dataset_url)

# Now, you can work with the 'df' DataFrame as needed
pd.set_option('display.max_columns', None)  # Show all columns
#print(df.head(10))
feature_columns = df[['curb-weight']]
feature_columns2 = df[['engine-size']]
target_variable_column = df[['city-mpg']]

# Convert selected columns to a NumPy array
X1_col = feature_columns.values
X2_col = feature_columns2.values
y_col = target_variable_column.values


#add X0 = 1 to each instance
X_new = np.array([[-2],[5]])
X_new_b = np.c_[np.ones((2,1)),X_new]

scale = StandardScaler()
X1_col= scale.fit_transform(X1_col)
X2_col=scale.fit_transform(X2_col)

#adding the BIAS to term X
X_b = np.c_[np.ones((len(X1_col),1)),X1_col]
X_b2 = np.c_[np.ones((len(X2_col),1)),X2_col]

#Hyperparameters
alpha = 0.1

#learning schedule hyperparameters
epochs = 300
t0,t1 = 5,50

def learning_schedule(t):
    return t0/(t+t1)


np.random.seed(42)
theta = np.random.randn(2,1)

#Storing theta
theta_path_sgd = []

m = len(X_b)
n_shown = 20

#Ploting Curb-Weight and City MPG
for nepoch in range(epochs):
    for iteration in range(m):
        if nepoch == 0 and iteration < n_shown:
            y_predict = X_new_b * theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration/n_shown+0.15))
            plt.plot(X_new,y_predict,color=color)

        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index +1]
        yi = y_col[random_index:random_index + 1]

        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(nepoch * m + iteration)
        theta = theta - eta * (gradients + 2 * alpha * theta) #L2
        theta_path_sgd.append(theta)


plt.plot(X1_col,y_col, "b.")
plt.xlabel("$Curb-Weight$")
plt.ylabel("$ City-MPG $")
plt.show()

print("\n")

#Ploting Engine Size and City MPG
for nepoch in range(epochs):
    for iteration in range(m):
        if nepoch == 0 and iteration < n_shown:
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(iteration / n_shown + 0.15))
            plt.plot(X_new,y_predict,color = color)

        random_index = np.random.randint(m)
        xi = X_b2[random_index:random_index + 1]
        yi = y_col[random_index:random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(nepoch * m + iteration)
        theta = theta - eta * (gradients + 2 * alpha * theta)
        theta_path_sgd.append(theta)

plt.plot(X2_col,y_col,"b.")
plt.xlabel("$Engine Size$")
plt.ylabel("$ City MPG $")
plt.show()