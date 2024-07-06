import lixirnet as ln
import matplotlib.pyplot as plt
import seaborn as sns


hodgkin_huxley = ln.HodgkinHuxleyModel()

history = {'voltage' : []}
for i in range(10000):
    hodgkin_huxley.iterate(i=30)
    history['voltage'].append(hodgkin_huxley.current_voltage)

sns.set_theme(style='darkgrid')

plt.plot(history['voltage'])
plt.show()
