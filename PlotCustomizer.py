import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
import tables
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from numpy import diff
from itertools import repeat
from functools import partial
import freud
from MDAnalysis import transformations
import ase
from IPython.display import display
from ipywidgets import IntProgress
import time
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scienceplots

# import tables
# from scipy.ndimage import gaussian_filter1d
# from scipy.interpolate import interp1d
# from numpy import diff
# from itertools import repeat
# from functools import partial
# import freud
# from MDAnalysis import transformations
# import ase
# from IPython.display import display
# from ipywidgets import IntProgress
# import time
# # from mpl_toolkits.mplot3d import Axes3D
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import scienceplots

plt.style.use(['science', 'ieee'])
plt.style.use('science')

# matplotlib
default_params = {'font.family': 'Times',
                  'font.serif': ["Times"],
                  # 'font.style':'normal',
                  # 'font.weight':'normal',
                  'font.size': 15,
                  # 'figure.dpi': 300,
                  # 'figure.edgecolor': 'white',
                  # 'figure.facecolor': 'white',
                  # 'figure.figsize': [6.4, 4.8],
                  # 'mathtext.default': 'regular', ## 上下标/math类text字体
                  # 'xtick.labelsize':5
                  }
default_params = {"text.usetex": False,
                  # 'font.family': 'Microsoft YaHei',
                  'font.family': 'Times New Roman',
                  'mathtext.fontset': 'custom',
                  'mathtext.rm': 'Times New Roman',
                  'mathtext.it': 'Times New Roman:italic',
                  'mathtext.bf': 'Times New Roman:bold',
                  # 'mathtext.default': 'regular', ## 上下标/math类text字体
                  'font.style':'normal',
                  # 'font.weight':'bold',
                  'font.size': 15,
                  # 'figure.dpi': 300,
                  # 'figure.edgecolor': 'white',
                  # 'figure.facecolor': 'white',
                  # 'figure.figsize': [6.4, 4.8],
                  'figure.figsize': [8, 6],
                  # 'xtick.labelsize':5
                  }

def formatnum(x, pos):
    try:
        num = x
        if int(num) == float(num):
            num = int(num)
        if num < 0:
            num = abs(num)  # 转为正值
            negative = True
        else:
            negative = False

        if num >= 1e4:  # num >= 10000 则强制科学计数
            # print('3')
            h = '{n:.2e}'.format(n=num)
        elif 1e-2 <= num < 1e4:  # 0.0001 <num < 10000则保留两位有效小数
            # print('4')
            h = '{n:1.2f}'.format(n=num)
        elif 1e-3 <= num < 1e-2:  # 0.0001 <num < 10000则保留两位有效小数
            # print('5')
            h = '{n:1.3f}'.format(n=num)
        elif 0 < num < 1e-3:  # 0.0001 <num < 10000则保留两位有效小数
            # print('6')
            h = '{n:1.2e}'.format(n=num)

        else:  # num > 0
            h = num

        if negative:  # #恢复负值
            h = '-' + h
        # x=format(x, '.2f')
        # sci_str = '{:g}'.format(x)
        sci_str = h

        if 'e+' in sci_str:
            # print(sci_str)
            gg = str(sci_str).split('e+')  # 分离科学计数法的底和10的指数部分
            gg[-1] = "$^{" + str(int(gg[-1])) + "}$"  # 当指数绝对值小于10时，去除指数前的0
            gg = 'x10'.join(gg)
        elif 'e-' in sci_str:
            gg = str(sci_str).split('e-')
            gg[-1] = "$^{-" + str(int(gg[-1])) + "}$"
            gg = 'x10'.join(gg)
        else:
            gg = sci_str
    except:
        gg = x
    return gg


axis_num_formatter = FuncFormatter(formatnum)
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/fancytextbox_demo.html#sphx-glr-gallery-text-labels-and-annotations-fancytextbox-demo-py
rcParams.update(default_params)
