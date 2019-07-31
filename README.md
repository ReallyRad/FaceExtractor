# FaceExtractor

This code shows how to extract faces with their gender from an unstructured data collection into a structured one.
It makes use of code from [this example](https://www.learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/) for the gender extraction.

We then made use of the Python image processing library to crop out faces from the dataset and save each individual face found as an image file.
The information is added to a list of dictionaries, with a key for the file name the face was extracted from, as well as a key for the detected gender.

This code was tested on python 3.6, using Pillow 6.1.0, opencv-python 4.1.0.25

To execute this code simply run `python --input data` in the terminal.

We imagine two ways this code could be reformatted for better reuseability and readability, which hasn't been done for lack of time : 
- as a module to be used within a bigger program
  In that case, the code shouldn't be executed from the command line with argument parsing, but rather individual methods from a module could be called to perform a variety of functions.
  It should return an easier to parse dataset, using for instance pandas dataframes, and be architectured according to the general guidelines of the bigger project.
  
- as a command line tool
  In that case, we would keep and add command line argument parsing possibilities and output the data as a JSON file with the corresponding image files. 
  This way this program could be part of a bigger data processing pipeline, for instance for creating faces datasets for training our own models based on different datasets created by this data processing pipeline
  
Results : 
Faces are not always detected and added to the dataset. Parameters as the confidence threshold provided in the code or other OpenCV detection functionality like multiscale detection could be used to enhance the results.
In the current stage the program crashes on the last image of the provided dataset.
