xaverius777/diss

"To evaluate Speech Enhancement (SE) systems, the most common method is to conduct subjective listening tests. They are expensive both in terms of time and resources, so a number of objective perceptual metrics have been developed to bypass the need for them. These metrics' quality is measured by the correlation they have with human scores; the model which has the highest one is Deep Noise Suppression Mean Opinion Score (DNSMOS), published by Microsoft. However, it only works for a type of subjective test, Absolute Category Rating (ACR). In this dissertation, we use it as a building block of a wider architecture that predicts another type of subjective listening test, Comparative Category Rating (CCR). The architecture is called Comparative Category Rater (CC-Rater) and we built the test set to evaluate it on, which is composed of 200 pairs of noisy speech and enhanced speech files. CC-Rater's predictions have a correlation of 0.34 with the test set's human scores, which opens the door to the possibility of building a metric that maximises this correlation and greatly decreases the need for CCR subjective listening tests. It also proves that the high quality audio embeddings within DNSMOS enable it to work as a pretrained model that can be used as the base for SE automatic evaluation metrics."

In this repository, you can find:

-CC-Rater

    -The model itself.
    
    -The variables the DNSMOS block needs to work (weights)
    
    -The code infrastructure that was used to train and test the model.
    
    -The model's trained weights.
    
-Testset

    -The testset for CC-Rater.
    
    -The clean speech and noise files with which the noisy speech was built.
    
    -The noisy speech files and the correspondent enhanced speech files.
    
    -The code used to:
    
      -extract the clean speech files candidates out of the Mozilla's Common Voice corpus' version in Kaggle (out of these candidates, the 200 final clean speech files were manually selected)
      
      -make the noise files out of noise samples manually selected from Zapsplat.
      
      -mix the clean speech and the noisy to produce the noisy speech files introduce reverberation into some of the noisy speech files
      -generate the Qualtrics surveys used to extract the CCR scores.
      
      -process the Qualtrics surveys results to make a .csv file, which is the actual testset.
      
