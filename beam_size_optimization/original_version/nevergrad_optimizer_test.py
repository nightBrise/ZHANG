import nevergrad as ng
import numpy as np
import matplotlib.pyplot as plt
from OptimizationTestFunctions import Michalewicz, AckleyTest, Sphere, Rastrigin
import itertools
import nevergrad.common.typing as tp

def _make_style_generator() -> tp.Iterator[str]:
    lines = itertools.cycle(["-", "--", ":", "-."])  # 4
    markers = itertools.cycle("ov^<>8sp*hHDd")  # 13
    colors = itertools.cycle("bgrcmyk")  # 7
    return (l + m + c for l, m, c in zip(lines, markers, colors))


class NameStyle(tp.Dict[str, tp.Any]):
    """Provides a style for each name, and keeps to it"""

    def __init__(self) -> None:
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name: str) -> tp.Any:
        if name not in self:
            super().__setitem__(name, next(self._gen))
        return super().__getitem__(name)
name_style = NameStyle()

work = ['HullAvgMetaTuneRecentering', 'NoisyDE', 'CmaFmin2', 'Cobyla', 'TBPSA', 'MultiCMA', 'NGOpt15', 'NelderMead']

test_optimizer = ['RandomSearch', 'NelderMead', 'Powell', 'BO']
xx = 20
def square(x, x0=xx):
    value = x.values()
    x = np.array(list(value))
    # x = x - np.linspace(0, 5, len(x))
    jitter = min(abs(x).min(), 1)
    return sum(x**2)*abs((np.random.normal(0, 0.5)))

michalewicz = Michalewicz(10)
ackleyTest = AckleyTest(3)
sphere = Sphere(10)
rastrigin = Rastrigin(2)
def test(*args, **kwargs):
    print(args)
    print(kwargs)
    x = kwargs
    value = x.values()
    x = np.array(list(value))
    return (rastrigin(x)+100)*abs((np.random.normal(1, 0.05)))

# test = square
para = {f'x{i}': ng.p.Scalar(init=10, lower=-20, upper=20) for i in range(5)}
para = ng.p.Instrumentation(**para)

print('f0: ', test(*para.args, **para.kwargs))

logger = ng.callbacks.ParametersLogger('data/test.log', append=False, order=1)
# stop = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask >=1000)
# costom_call = lambda x,y,z:print(x, y, z)

optimizer = ng.optimizers.registry['TBPSA'](parametrization=para, budget=2000, num_workers=1)
optimizer.register_callback("tell", logger)
# optimizer.register_callback("tell", costom_call)
# optimizer.register_callback("ask", stop)

recommendation = optimizer.minimize(test, verbosity=1)
list_of_dict_of_data = logger.load()
print("recommendation ",recommendation.value)
print("loss", test(*recommendation.args, **recommendation.kwargs))


loss = []
for data in list_of_dict_of_data:
    loss.append(data['#loss'])
plt.plot(loss)

optimizer = ng.optimizers.registry['TBPSA'](parametrization=para, budget=1, num_workers=1)
recommendation = optimizer.minimize(test, verbosity=1)
# data = [square(*para.args, **para.kwargs) for i in range(1000)]
# plt.figure()
# plt.plot(data)
plt.show()
exit()
loss = {}

for name in test_optimizer:
    optimizer = ng.optimizers.registry[name]
    opt_loss = []
    print(name)
    for budget in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        # timer = ng.callbacks.EarlyStopping.timer(600)
        opt = optimizer(parametrization=para, budget=budget, num_workers=1)
        opt.register_callback("tell", logger)
        # opt.register_callback("ask", timer)
        recommendation = opt.minimize(test)
        opt_loss.append(test(*recommendation.args, **recommendation.kwargs))
    loss[name] = opt_loss

for key, value in loss.items():
    x=  [10, 20, 50, 100, 200, 500, 1000, 2000]
    plt.plot(x, value, name_style[key], label=key)

plt.ylabel('$f(x)$')
plt.xlabel('Iteration')
plt.grid(True, which="both")
plt.legend()
plt.savefig('data/test_opt.png', dpi=300)
plt.show()
