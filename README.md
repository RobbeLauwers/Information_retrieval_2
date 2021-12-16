# Information retrieval 2: Document Ranking Using a Neural Retrieval Model

Instead of running the model training on our own computers, we used Google Colab. The online notebook is available here: 
https://colab.research.google.com/drive/1c2DH0ytXTdclzNoDIaTNI2d_eU5XyDun?usp=sharing

It is view only, so you should take a copy to use it. 

If you want to run any code, first run the first two cells: the first will install a 
required library, the second will give Colab access to your Google Drive. The code has several hardcoded paths, and 
assumes that your drive has a folder "data" containing all datasets provided in the assignment. 

Most of the code cells are copy-pasted versions of our implementation with only a few parameter changes. We did this so 
that the output would be kept, but this does make the notebook difficult to read. The top line of a cell is a comment line specifying the parameters used. You should not 
have to run these cells again, as the output has been kept. For a more readable version of which tests we ran with 
which results, see test_results.txt in this repository.

The final two cells in the notebook are used to generate the final model. The next-to-last cell trains the model on the 
full dataset. The last cell generates output using this model. This output is available in this repository as 
full_output.csv. This is not yet the output asked for in the assignment: the documents are not yet sorted and the labels
are still floating point numbers.

To convert the output to the correct format, run sort_output.py in this repository. This will generate the 
neural_model_rank asked for in the assignment.

A copy of the Colab notebook is available in this repository as notebook.ipynb. However, we did not test if giving it 
access to Google Drive actually works outside Colab. It is provided purely for completeness in this repository.