## Experiments Explained:  

`standard`:  
Reference Run 

`color_jitter_rendered`:  
*Q1:* Does data augmentation for the rendered image help ?
*Q2:* Does data augmentation for the real image help ?

`crop_level`:  
*Q1:* How far should the bounding box be expanded to gather more scene context ?

`iterations`:  
*Q1:* How often should the flow estimate be refined ?

`non_synthetic_data`:  
*Q1:* Does the usage of synthetic data increase the generalization capability of the network ?

`noice_level`:  
*Q1:* How should the inital pose distriution sampled under the condution not fine-tuning to a specific inital pose network.

`nr_of_discrete_viewpoints`:  
*Q1:* Size of the viewpoint database.


