# %% 
from sklearn.tree import DecisionTreeRegressor
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np

# Define the dataset
# X = np.array([[1], [3], [4], [7], [9], [10], [11], [13], [14], [16]])
# y = np.array([3, 4, 3, 15, 17, 15, 18, 7, 3, 4])

X=np.arange(1,13,1).reshape(-1,1)
y=np.concatenate((np.arange(1,12,1),12), axis=None)

# X = np.array([[1, 2], [3, 4], [4, 5], [7, 2], [9, 5], [10, 4], [11, 3], [13, 5], [14, 3], [16, 1],
#               [10, 10], [16, 10], [12, 10]]).reshape(-1,2)
# y = np.array([3, 4, 3, 15, 17, 15, 18, 7, 3, 4,8,10,13]).reshape(-1,1)


# Fit the decision tree model
max_depth = 2
model = DecisionTreeRegressor(max_depth=max_depth)
model.fit(X, y)

# Generate predictions for a sequence of x values
x_seq = np.arange(0, 17, 0.1).reshape(-1, 1)
y_pred = model.predict(x_seq)
# %%


# Plot the dataset with the decision tree splits
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue')
plt.plot(x_seq, y_pred, color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression (max_depth=%d)' % max_depth)
plt.show()
# %%
# Create an interactive 3D plot with Plotly



model = DecisionTreeRegressor(max_depth=2)
model.fit(X, y)

# Generate predictions for a sequence of x values
x_seq = np.arange(0, 17, 0.1).reshape(-1, 1)
y_pred = model.predict(X)


x_seq = np.arange(0, 18, 0.25)
y_seq = np.arange(0, 18, 0.25)
z_seq = np.array([model.predict(np.array([[x, y]])) for y in y_seq for x in x_seq]).reshape(len(x_seq), len(y_seq))
                                                                
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])

fig.add_trace(go.Surface(x=x_seq, y=y_seq, z=z_seq, colorscale='Viridis', showscale=True, opacity = 0.75),
            row=1, col=1)

fig.add_trace(go.Scatter3d(x=X[:, 0], y=X[:, 1], z=y.flatten(), mode='markers', marker=dict(size=5, color='red')),
            row=1, col=1)

fig.update_layout(title='Decision Tree with Max Depth = {}'.format(max_depth),
                scene=dict(xaxis_title='x1', yaxis_title='x2', zaxis_title='Predicted Y'),
                autosize=False, margin=dict(l=5, r=50, b=25, t=60),
                width=800, height=600
                )
    

fig.show()
# %%
