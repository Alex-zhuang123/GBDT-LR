# GBDT+LR融合搭建风控模型 

variable_bin_methods.py:分箱函数

variable_encode.py：编码函数

feature_selector.py：特征筛选函数

GBDT+LR.py：主函数

利用GBDT模型进行特征衍生，将衍生特征和原数据合并，筛选出高价值特征输入LR模型做进一步风险预测。
