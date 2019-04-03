# 1.11 Types

Important is the distintion between types and tokens. University as a word
is a token, whereas all such instances are a type of university. Mostly, 
linguistics focuses on working with types with respect to their
morphological properties and various forms. 

How can statistical reasoning be used when dealing with types?
Statistical approaches often make an independence assumption, 
which, in most cases, isn't valid for language as words, sentences, 
documents are highly interdependent. 

# 1.12 Why linguistic structure is a moving target

Language changes over time, linguistic theories emerge and some
make it to computational approaches (such as frame semantics). Often times
creation of corpora sparks interest in a subject. 

# 1.13 Conclusion

Lingustic structure predictions boils down to computationally 
solving ambiguity which humans deal naturally. Doing lingustic structure
prediction can be divided into three steps:

1. Define inputs and outputs
2. Define a scoring/evaluation function over input-output pairs, 
and an algorithm to maximize the scoring function
3. Apply some algorithm to tune the parameters of the scoring function
4. Evaluate the results using an objective criterion

