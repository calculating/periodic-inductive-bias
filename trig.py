#%%
import torch as t

import matplotlib.pyplot as plt
# %%
test = t.rand(16, 64) * 2
freq = t.rand(16, 1) * 5
amp = t.rand(16, 1) * 5
sine = amp * t.sin(freq * test)
# %%
# for i in range(16):
#     plt.plot(test[i], sine[i], 'o')
plt.plot(test, sine, 'bo')
plt.show()
# %%
