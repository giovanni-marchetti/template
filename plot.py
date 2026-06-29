import numpy as np
import matplotlib.pyplot as plt
import os


train_errors = np.load(os.path.join('results', 'train_errors.npy'))
test_errors = np.load(os.path.join('results', 'test_errors.npy'))


plt.figure()
plt.plot(np.arange(len(train_errors)), train_errors, label='train', linewidth=2)
plt.plot(np.arange(len(test_errors)), test_errors, label='test', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.savefig(os.path.join('results', 'errors.png'))
plt.show()

