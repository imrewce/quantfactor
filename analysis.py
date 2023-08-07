from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
from qlib.workflow import R


import qlib
from qlib.tests.config import CSI300_BENCH, CSI300_GBDT_TASK


if __name__ == "__main__":

  data_uri = "~/.qlib/qlib_data/cn_data" 
  qlib.init(provider_uri=data_uri, region='cn')
  
  
  model = init_instance_by_config(CSI300_GBDT_TASK["model"])
  dataset = init_instance_by_config(CSI300_GBDT_TASK["dataset"])
  
  recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="workflow")
  print(recorder)
  pred_df = recorder.load_object("pred.pkl")
  
  report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
  positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
  analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
  
  analysis_position.report_graph(report_normal_df)
  analysis_position.risk_analysis_graph(analysis_df, report_normal_df)
  #Analysismode
  label_df = dataset.prepare("test", col_set="label")
  label_df.columns = ["label"]
  
  pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
  analysis_position.score_ic_graph(pred_label)
  #Model performance
  analysis_model.model_performance_graph(pred_label)
