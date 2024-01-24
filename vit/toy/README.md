Location: /users/PAS1043/osu7903/work/vit/toy

Purpose:
Idea is to generate toy images to test vision transformers for 
both classification and regression.
Toy images will be 100pixel x 100pixel images, with a line of
random slope in the middle.  Points along the line are sampled.
Add in random noise throughout the image.

Code:
- Generation of toy images
  All of the images generated are 50 samples of points along a line of random slope.
  We add in 25 points of noise for the "low" noise samples, 
      and 100 points of noise for the "hig" noise samples.
  a) images for classification
     - code is: generate_images_classes.py
     - Here we generate two classes of line:
          Positive slope (angles from 10-40 degrees)
          Negative slope (angles from 140-170 degrees)
          These are placed in subirs under /classification
  b) images for regression
    - code is: generate_images.py (and generate_images_high_noise.py)
     - Here we generate lines of any random slope from 0-180 degrees
          These are placed in subirs under /regression

- Classification
  Use the repo: https://github.com/lucidrains/vit-pytorch
  Code is: train_vit_classification.py
  Scripts are: 
       run_test_vit_toy_classification_low_noise.sh
       run_test_vit_toy_classification_high_noise.sh

- Regression
  Use the repo: https://github.com/lucidrains/vit-pytorch
  Code is: train_vit_regression.py
     ==> Note modifications needed to get regression to work!
  Scripts are: 
       run_test_vit_toy_regression_low_noise.sh
       run_test_vit_toy_regression_high_noise.sh


