# BloodGlucosePrediction
We developed a Hierarchical Multi-Factor Fusion Framework for blood glucose prediction. It adaptively switches between Transfer LSTM, GRU, and Bi-LSTM by coupling CGM, GI, and Insulin data. This architecture effectively resolves model lag and cold-start data scarcity.

# Project Introduction
  This project presents a novel Hierarchical Multi-Factor Fusion Framework for proactive blood glucose prediction, specifically designed to address the challenges of data scarcity in "cold-start" scenarios and the inherent temporal inertia of traditional models. Unlike conventional autoregressive approaches that rely solely on Continuous Glucose Monitoring (CGM) data, our system integrates multi-dimensional physiological factors, including Glycemic Index (GI) impact and Insulin pharmacokinetics, into a deep learning architecture.
  The core innovation lies in a Dynamic Stratified Scheduling Strategy. By leveraging a Genetic Algorithm (GA) to optimize data-volume thresholds, the system adaptively switches between a Frozen-Transfer LSTM for small datasets, a lightweight GRU for transitional phases, and a full-scale Bi-LSTM for long-term steady states. Experimental results on a 45-minute prediction horizon demonstrate that our model significantly mitigates the "phase lag" common in baseline models, reducing temporal latency by up to 50% in large-scale datasets. Furthermore, the clinical safety of the framework is validated through Clarke Error Grid Analysis, with over 98% of predictions falling within the clinically acceptable Zones A and B. This research provides a robust, clinically-oriented solution for personalized diabetes management, shifting the paradigm from reactive monitoring to proactive physiological intervention.






  
