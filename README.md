# Human Activity Recognition (HAR)

Applications of inertial sensors span both commerical and military uses, from tactical navigation and unmanned aerial vehicles (UAVs), to underwater and space exploration, aggressive driving detection, and health and wellness trackers. Physical inactivity increases the risk of adverse health conditions like coronary heart disease, reduced blood circulation and clotting, type-2 diabetes, and can simply shorten one's life expectancy. Machine Learning (ML) models that leverage inertial data can aid in the monitoring and encouraging of physical activity. This can be especially important amongst at risk and ageing populations, where physical activity can reduce chronic diseases and lower the risk of developing certain conditions. 

#### Purpose:
This repository contains source code and models used to prove whether raw inertial data from modern-day smartphones can be used to train and evaluate a machine learning model and predict low-intensity human activity with effective accuracy. The alternative can be quite complex and time consuming, requiring domain knowledge, signal filtering, feature extraction, feature selection, and normalization; all this before even performing model selection, training, evaluation and optimization. 

#### Dataset: 
The dataset used is multivariate, time-series data. It contains tri-axial data from an accelerometer and gyroscope. The data is from 30 volunteer subjects randomly split into 70% training (21 subjects) and 30% testing (7 subjects). Each data point corresponds to one of six activities of daily life (ADL) including:

* Walking
* Walking Upstairs
* Walking Downstairs
* Sitting
* Standing
* Laying

Public dataset creators: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [UCI Machine Learning Repository](https://doi.org/10.24432/C54S4K)
