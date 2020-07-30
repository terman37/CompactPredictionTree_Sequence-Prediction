# Compact Prediction Tree - Sequence Prediction

An example of sequence prediction based on the CPT algorithm

Docs available [here](/CPT_docs/CPT ADMA2013_Compact_Prediction_trees.pdf) and [here](/CPT_docs/CPT+1167-Article Text-2062-1-10-20171231.pdf)



Main part of the algorithm is based on CPT implementation from here https://github.com/NeerajSarwan/CPT

Tweaked for my own needs.



#### DataSet:

- Consisting of over 44k sequences of different lengths (from 2 items to 10) reprensenting over129k items
- there exists 370 differents CODES
- train.csv file desciption
  - ID: gouping id for sequence
  - CODE: sequence item 
  - LINE_NB: poistion of item in sequence



#### Target:

Predict the last item in sequence.



#### Training:

Training the model consist of building the Tree, Inverted index and Lookup Table

- Tree: hierachical tree modeling the sequences

  <img src="images\tree_sample.png" alt="tree_sample" style="zoom: 80%;" />

- Inverted Index: dictionnary giving in which sequence each code is used

  here CODE: 'PX9' is used in sequences 8,1,4,5

![image-20200730141530242](images\II.png)

- Lookup Table: dictionnary giving node adress of last element of a sequence:

  ![LT](images\LT.png)

  

#### Predictions:

Concept, For a given sequence:

- find all sequences containing any its item using Inverted Index
- Rebuild the original sequences using the Lookup Table (avoiding to save the original data)
- then for each original similar sequence:
  - find position corresponding to the last item in the sequence to predict
  - calculate a score for each possible following item (check docs for global description)

- return the n elements with biggest score

  

#### API:

Simple API using FastAPI and can be run Docker

<img src="images\postman.png" alt="postman" style="zoom: 80%;" />

#### Next steps:

Due to nature of data, we can see that only the 2 or 3 preceding items are important. Thus, I decided to switch to a more classical approach using Decision tree / Random Forest calssifiers. it also permits more flexible approach allowing additional input feature to improve model performance.