import matplotlib.pyplot as plt
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


log_dir = os.path.join('results', 'tensorboard')


def load_scalar(accumulator, tag):
    events = accumulator.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


accumulator = EventAccumulator(log_dir)
accumulator.Reload()

train_steps, train_errors = load_scalar(accumulator, 'error/train')
test_steps, test_errors = load_scalar(accumulator, 'error/test')


plt.figure()
plt.plot(train_steps, train_errors, label='train', linewidth=2)
plt.plot(test_steps, test_errors, label='test', linewidth=2)
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend()
plt.savefig(os.path.join('results', 'errors.png'))
plt.show()
