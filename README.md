# Digital-PhotoMonatge (Python)

Graphical user interface Application to perform manual and automatic graph cut composites of images

# To install
  - Clone the repo
  - run `conda env create -f environment.yml`
  - Activate your conda environment using `conda activate photoMontage`

# To run example
  - run `python GUI.py` from your terminal
  - Load images from resources folder (famille1.jpg ...) using `Load Image x`
  - Click on `AutoMontage` button
  - Perform any corrections by drawing over an image or removing a mask after `autoMontage` has finished
  - If any corrections were done, click on `GraphCut` button
  
# Features
  - For graph cut using manual context, draw on picture and press `GraphCut`
  - You can take pictures from webcam by pressing `1`, `2`, `3`, `4` on your keyboard
  - You can erase drawn masks by pressing `e` and entering erase mode. Pressing `e` again will remove erase mode
  - You cna save images taken by pressing the `Save images` button

