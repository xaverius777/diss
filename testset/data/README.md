Directory xaverius777/diss/testset/data contains the following:
+ Noise - the noise files used for the noisy speech. They were made with audio_catter.py from noises manually selected from Zapsplat: https://www.zapsplat.com
+ Clean speech - the clean speech files used for the noisy speech. An initial set of candidate files was first extracted from Mozilla's Common Voice dataset on Kaggle ( https://www.kaggle.com/datasets/mozillaorg/common-voice), I used audio_fetcher.py for that. The final clean speech files were manually selected from this set. 
+ Noisy speech - made by giving the noise and clean speech files to file_mixer.py
+ Enhanced speech - made by passing the noisy speech through Microsoft's NSNet 2 model: https://github.com/microsoft/DNS-Challenge/tree/master/NSNet2-baseline
