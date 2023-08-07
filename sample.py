import qlib
#from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


if __name__ == "__main__":

 data_uri = 'E:/qlib/qlib_data/cn_data'
 qlib.init(provider_uri=data_uri, region=REG_CN)
  
 model = init_instance_by_config(CSI300_GBDT_TASK["model"])
 dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
 port_analysis_config = {
 "executor": {
 "class": "SimulatorExecutor",
 "module_path": "qlib.backtest.executor",
 "kwargs": {
 "time_per_step": "day",
 "generate_portfolio_metrics": True,
            },
        },
 "strategy": {
 "class": "TopkDropoutStrategy",
 "module_path": "qlib.contrib.strategy.signal_strategy",
 "kwargs": {
 "signal": (model, dataset),
 "topk": 50,
 "n_drop": 5,
            },
        },
 "backtest": {
 "start_time": "2017-01-01",
 "end_time": "2020-08-01",
 "account": 100000000,
 "benchmark": CSI300_BENCH,
 "exchange_kwargs": {
 "freq": "day",
 "limit_threshold": 0.095,
 "deal_price": "close",
 "open_cost": 0.0005,
 "close_cost": 0.0015,
 "min_cost": 5,
            },
        },
    }
 # NOTE: This line is optional
 # It demonstrates that the dataset can be used standalone.
 example_df = dataset.prepare("train")
 print(example_df.head())
 # start exp
 with R.start(experiment_name="workflow"):
 R.log_params(**flatten_dict(CSI300_GBDT_TASK))
 model.fit(dataset)
 R.save_objects(**{"params.pkl": model})
 # prediction
 recorder = R.get_recorder()
 sr = SignalRecord(model, dataset, recorder)
 sr.generate()
 # Signal Analysis
 sar = SigAnaRecord(recorder)
 sar.generate()
 # backtest. If users want to use backtest based on their own prediction,
 # please refer to https://qlib.readthedocs.io/en/latest/component/recorder.html#record-template.
 par = PortAnaRecord(recorder, port_analysis_config, "day")
 par.generate()

'''
provider_uri="~/.qlib/qlib_data/cn_data" # on 244 
#under the path of qlib: python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
qlib.init(provider_uri,region='cn') #REG_CN


data_handler_config = {
 "start_time":"2020-01-01",
 "end_time":"2020-11-30",
 "fit_start_time":"2020-01-01",
 "fit_end time":"2020-06-30",
 "instruments":"all"
        }

 h =Alpha158(**data_handler_config)
 # 获取列名(因子名称)/标签名
# print(h.get_cols0())/ Alpha158_df_label = h.fetch(col_set="label")
'''

