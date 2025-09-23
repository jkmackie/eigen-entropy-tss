<p align="center">
    <img src="https://github.com/jkmackie/eigen-entropy-tss/blob/main/images/total_recall1.png"/>
</p>

# eigen-entropy-tss coded by [jkmackie](https://github.com/jkmackie)

Python implementation of the Eigen-entropy based Time Series Signatures algorithm (EE - TSS) by [jkmackie](https://github.com/jkmackie).  To my knowledge, this is the first public implementation of EE - TSS.

The technique "achieves high recall rates with limited clinical datasets but also ensures the algorithm's feature generation is understandable, addressing a critical need for clinician-friendly tools."  Also, it cuts the dimensions of high-dimensionality data by completely transforming it.  The low dimension data--where the number of dimensions equals the number of scale factors--can then be fed into the classifier model.

The algorithm requires computational resources.  It is coded in Python with Joblib parallel processing.  

The binary classification dataset [heartbeat](https://www.timeseriesclassification.com/description.php?Dataset=Heartbeat) is used to illustrate the algorithm.  The classes are: 0=Normal and 1=Abnormal.  Each heartbeat sample is a 405 observation time series.  We must catch all abnormal heartbeats so recall is the key metric.

**EE - TS results with heartbeat:**
* For the original heartbeat data, the Ridge Classifier returns Recall = 98.64%
* EE - TSS transformed data and Ridge Classifier achieves Recall = 100%

<br><br>
$\color{#00ff00}{\textsf{TERMS OF USE:  MIT No AI License}}$

<br><br>
**Citations:**
```bibtex
@code{
    author={jkmackie},
    title={eigen-entropy-tss},
    repo={https://github.com/jkmackie/eigen-entropy-tss},
    year={2025}
}
```
```bibtex
@article{
    author={Patharkar, A., Huang, J., Wu, T. et al.},
    title={Eigen‑entropy based time series signatures to support multivariate time series classification},
    year={2024},  
    journal={Nature Scientific Reports},
    url= {https://www.nature.com/articles/s41598-024-66953-7},
    doi={10.1038/s41598-024-66953-7}
}
```
