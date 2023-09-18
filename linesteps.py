# %%
import wandb
import random
import torch as t

# %%

wandb.init(
    # set the wandb project where this run will be logged
    project="timeline"
)
# %%

# simulate training
epochs = 10
offset = random.random() / 5
x_points = t.rand(16)
y_points = []
y_keys = []
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    y_points.append((loss * x_points).numpy())
    y_keys.append(str(loss))

wandb.log({'multiline': wandb.plot.line_series(
    xs=x_points.numpy(),
    ys=y_points,
    keys=y_keys,
    title="muliline over time",
    xname="x points")})
    
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()

# %%
testrand = t.rand(16)
testy = []
for i in range(10):
    testy.append((testrand * i).numpy())
print(len(testrand.numpy()))
print(len(testy))
print(len(testy[0]))

# %%
print(x_points.shape())

# %%
