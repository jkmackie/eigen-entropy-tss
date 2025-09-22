
![Total Recall Schwarzenegger](./images/total_recall1.PNG)


# eigen-entropy-tss
Full implementation of the Eigen-entropy based Time Series Signatures algorithm (EE - TSS).  The technique "achieves high recall rates with limited clinical datasets but also ensures the algorithm's feature generation is understandable, addressing a critical need for clinician-friendly tools."

The algorithm is coded in Python with Joblib parallel processing.  The binary classification dataset heartbeat is used to illustrate the algorithm.  The classes are 0=Normal and 1=Abnormal.  We want to catch all abnormal heartbeats so a recall is the key metric.

* For the original heartbeat data, a Ridge Classifier returns Recall = 98.64%
* EE - TSS transformed data and Ridge Classifier achieves Recall = 100%

**Citation:**
```bibtex
@article{
    author={Patharkar, A., Huang, J., Wu, T. et al.},
    title={Eigen‑entropy based time series signatures to support multivariate time series classification},
    year={2024},  
    journal={Sci Rep},
    url= {https://www.nature.com/articles/s41598-024-66953-7},
    doi={10.1038/s41598-024-66953-7}
}
