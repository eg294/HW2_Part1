import pandas as pd
import matplotlib.pyplot as plt

#===============================Question 1===============================
# Replace 'imports-85.csv' with the actual file path if it's not in the current directory
dataset_url = "/home/eg294/Documents/CS370/imports-85.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(dataset_url)

# Now, you can work with the 'df' DataFrame as needed
pd.set_option('display.max_columns', None)  # Show all columns
#print(df.head(10))
feature_columns = df[['curb-weight', 'engine-size']]
target_variable_column = df[['city-mpg']]


# Convert selected columns to a NumPy array
X_col = feature_columns.values
y_col = target_variable_column.values

#Plot layout
#plt.title("Predicted vs Curb Weight") 
#plt.xlabel("$curb-weight$")
#plt.ylabel("$ engine-size $")
#plt.scatter(X[:,0],y)
#plt.show()



def gradient_descent(m_now, b_now, xpoints,ypoints, L):

    m_gradient = 0
    b_gradient = 0

    n = len(xpoints)

    for i in range(n):
        x = xpoints[i]
        y = ypoints[i]

        m_gradient += -(2/n)* x * (y -(m_now * x * b_now))
        b_gradient += -(2/n) * (y -(m_now * x * b_now))


    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b

#===============================
#X = from the curb weight
#y = miles per gallon  (Dependent from X )
m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    m,b = gradient_descent(m,b,X_col,y_col,L)

print(m,b)

plt.title("Predicted vs Curb Weight") 
plt.xlabel("$curb-weight$")
plt.ylabel("$ City MPG $")
plt.scatter(X_col[:,0],y_col,color="blue")
plt.plot([m * x + b for x in X_col[:,0]],color = "red")
plt.show()


#===============================
#X = from the Engine Size
#y = Miles per gallon  (Dependent from X )
m1 = 0
b1 = 0
L1 = 0.0001
epochs1 = 1000

for i in range(epochs1):
    m1,b1 = gradient_descent(m1,b1,X_col[:,1],y_col,L1)

print(m1,b1)

plt.title("Predicted vs Engine Size") 
plt.xlabel("$engine-size$")
plt.ylabel("$ City MPG $")
plt.scatter(X_col[:,1],y_col,color="blue")
plt.plot([m1 * x + b1 for x in X_col[:,1]],color = "red")
plt.show()