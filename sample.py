import empyrical as ep
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.workflow import R  # 实验记录管理器
# from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord, SignalRecord
from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import DatasetH
from qlib.data.dataset.processor import DropnaLabel, ProcessInf, CSRankNorm, Fillna
# from qlib.utils import init_instance_by_config
from typing import List, Tuple, Dict

'''
from scr.core import calc_sigma, calc_weight
from scr.factor_analyze import clean_factor_data, get_factor_group_returns
from scr.qlib_workflow import run_model
from scr.plotting import model_performance_graph, report_graph
'''

import matplotlib.pyplot as plt
import seaborn as sns

# plt中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
# plt显示负号
plt.rcParams["axes.unicode_minus"] = False
qlib.init(provider_uri="qlib_data", region="cn")
