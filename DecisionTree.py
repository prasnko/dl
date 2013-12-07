#-------------------------------------------------------------------------------
# Name:        OR Tree Deep Learning
# Purpose:     Deep Learning Research Project
#
# Author:      PRASANNA
#
# Created:     04/02/2013
# Copyright:   (c) PRASANNA 2013
# Licence:     GPL
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# Data Structures and Functions
#
# splitDataset -> Splits the dataset based on the attribte index and it's value both of which are passed to it and returns the sub dataset.
# newFeatVec stores a list of features without the attribute value that is matched.
# newX stores the reduced dataset.
#
# getBestFeature -> Returns the feature that provides the maximum increase in entropy.
#
# featList stores the feature list for each attribute.
# subDataset stores the reduced dataset that is returned by SplitDataset function
#
# createTree -> It is used to create the decision tree. It return the decision tree that is created in the form of a dictionary.
# decisionTree is a dictionary that is used to build the decision Tree recursively by storing another dictionary for next level.
# labels stores a list of attribute names.
# level variable helps in formating the decision tree based on the level of the recursive call
#
#
# formatText -> Formats the text document that is passed to it to return two lists:
# words: which is training data in the form of list of lists.
# attributes: which is a list of names of the features.
#
# pruneUniformDistributions -> After entire decision tree has been  created we traverse the tree bottom up to remove nodes that are uniformly distributed.
#
# removeIndependentNodes -> After entire decision tree has been  created we traverse the tree bottom up to remove nodes that are independent of the rest of the tree network given some other node.
#-------------------------------------------------------------------------------


from math import log
import operator
import re
import sys
from sys import stdout
import numpy
import matplotlib.pyplot as plt
import pylab as pl
import pydot
import dot_parser
import random

def calcEntropy(dataSet,index):
 numEntries = len(dataSet)
 labelCounts = {}
 # Classifying each feature vector in the dataset based on the class label
 for featVec in dataSet:
  currentLabel = featVec[index]
  if currentLabel not in labelCounts.keys():
   labelCounts[currentLabel] = 0
  labelCounts[currentLabel] += 1
 entropy = 0.0
 # Calculating probability and then the entropy
 for key in labelCounts.keys():
  prob = float(labelCounts[key])/numEntries
  entropy -= prob * log(prob,2)
 return entropy

def splitDataSet(X, attrIndex, value):
 newX = []
 for featVec in X:
  # Keeping only those feature vectors whose feature value matches the required but removing the value from the vector.
  if featVec[attrIndex] == value:
   newFeatVec = featVec[:attrIndex]
   newFeatVec.extend(featVec[attrIndex+1:])
   newX.append(newFeatVec)
 return newX

def getBestFeature(dataSet):
 numFeatures = len(dataSet[0]) - 1
 ##entropy = 
 calcEntropy(dataSet)
 ##bestInfoGain = 0.0;
 bestEntropy = 0.0
 bestFeature = -1
 bestValueCounts = []
 # Check each feature in the dataset, if it's entropy is highest.
 for i in range(numFeatures):
  valueCounts = []
  featList = [example[i] for example in dataSet]
  uniqueVals = set(featList)
  newEntropy = 0.0
 # For each feature value, split the dataset on that value and calculate the entropy on it.
  for value in uniqueVals:
   subDataSet = splitDataSet(dataSet, i, value)
   prob = len(subDataSet)/float(len(dataSet))
   valueCounts.append(len(subDataSet))
   newEntropy -= prob * log(prob,2)
# Highest entropy gain providing feature is the best attribute. 
# (Note: We are trying to generate a balanced decision tree and so entropy (and not information gain) is to be maimized.
  if (newEntropy > bestEntropy):
    bestEntropy = newEntropy
    bestFeature = i
    bestValueCounts = valueCounts
 return bestFeature,bestValueCounts


totalCount =0

def createTree(dataSet,labels,wholeClassList,wholeDataSet,level,tc):
 global totalCount,maxLevel
 totalCount = tc
 level+=1
 if level>maxLevel:
    maxLevel = level
 if len(dataSet)>0:
 ## If dataset has been split to a single feature then return the number of examples that have been classified along that path of the decision tree.
  if len(dataSet[0]) == 1:
     return len(dataSet)
 bestFeat, bestValueCounts = getBestFeature(dataSet)
 bestFeatLabel = labels[bestFeat]
 # Store decision tree recursively in dictionary.
 decisionTree = {bestFeatLabel:{}}
 del(labels[bestFeat])
 featValues = [example[bestFeat] for example in wholeDataSet]
 # Extracting unique value of features for given attribute.
 uniqueVals = set(featValues)
 # For each value of the best feature selected, generate the tree.
 for value in uniqueVals:
  treeOutput = ''
  if level==1:
    stdout.write('\n')
  for i in range(0,level-1):
    if i==0 :
     stdout.write("\n")
    treeOutput += '| '
  treeOutput += bestFeatLabel+'='+str(value)
  stdout.write("%s" %treeOutput)
  subLabels = labels[:]
  decisionTree[bestFeatLabel][value] = createTree(splitDataSet\
(dataSet, bestFeat, value),subLabels,wholeClassList,wholeDataSet,level,totalCount)
 # If value returned from lower level is a number return it as sample count.
  if type(decisionTree[bestFeatLabel][value]).__name__ != 'dict':
     stdout.write(":%d" % int(decisionTree[bestFeatLabel][value]))
     totalCount += decisionTree[bestFeatLabel][value]
 return decisionTree

# Classifying one feature vector at a time

def formatText(x):
    lines = []
    attributes = []
    words = []
    text = x.read()
    lines = text.split('\n')
    # spliting text into a list of words
    attr_name = re.split('\W+',lines[0])
    i=0
    # Extracting the attribute names
    for attr in attr_name:
      if i%2==0:
       attributes.append(attr)
      i= i+1
    # Removing the attribute name from training data.
    lines.remove(str(lines[0]))
    lines.remove('')
    # Storing each feature vector in a list.
    for line in lines:
     words.append(line.split('\t'))
    return words,attributes


def pruneUniformDistributions(tree,counts,match,removeUniform):
 for key in tree.keys():
  if type(tree[key]).__name__ == 'dict':
    pruneUniformDistributions(tree[key],counts,match,removeUniform)
    if len(counts)== 2:
     if removeUniform[0]>0:
        tree[key]= removeUniform[0]
        removeUniform[0]=0
     if match[0]>0:
        removeUniform[0]=match[0]
        match[0]=0
     counts=[]
  else:
    counts.append(tree[key])
    if len(counts)==2:
        if abs((counts[0])-(counts[1]))<=2:
         match[0]=(counts[0])+(counts[1])
        else:
         counts=[]
 return tree

def removeIndependentNodes(tree,counts,prune,removeIndependent,currentNode):
 for key in tree.keys():
  if type(tree[key]).__name__ == 'dict':
    removeIndependentNodes(tree[key],counts,prune,removeIndependent,currentNode)
    if len(counts)==4:
     if currentNode[0]>0:
      print tree[key]
      print '\n'
      tree[key]['1']= currentNode[0]
      print  currentNode[0]
      print '\n'
      print  currentNode[1]
      tree[key]['0']= currentNode[1]
      currentNode[0]=0
      print tree[key]
      print '\n'
     if removeIndependent[0]>0:
      currentNode[0]=removeIndependent[0]
      currentNode[1]=removeIndependent[1]
      removeIndependent[0]=0
     if prune[0]>0:
      removeIndependent[0]=prune[0]
      removeIndependent[1]=prune[1]
      prune[0]=0
     counts=[]
  else:
    counts.append(tree[key])
     if (abs((counts[0])-(counts[2]))<=4 and abs((counts[1])-(counts[3]))<=4):
        prune[0] = (counts[0])+(counts[2])
        prune[1] = (counts[1])+(counts[3])
     else:
      counts=[]
 return tree


iid = 1
def next_id():
    global iid
    res = iid
    iid += 1
    return res

# Drawing an edge between the two nodes.
def draw(graph,node_a,node_b):
     graph.add_edge(pydot.Edge(node_a, node_b))

# Visiting each node recursively to print it.
def visit(graph,node,count,level,parent=None):
    level+=1
    for k,v in node.iteritems():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent!=None:
             if level==2:
              a = -2
             elif k=='0':
              a = (count-1)
             else:
              a = (count)
             node_a = pydot.Node(a,label=parent, style="filled", fillcolor="red")
             graph.add_node(node_a)
             b= (next_id())
             node_b = pydot.Node(b,label=k, style="filled", fillcolor="green")
             graph.add_node(node_b)
             draw(graph,node_a, node_b)
             count=b
            visit(graph,v,count,level,k)
        else:
            if k=='0':
              a = (count-1)
            else:
              a = (count)
            node_a = pydot.Node(a,label=parent, style="filled", fillcolor="red")
            graph.add_node(node_a)
            b= (next_id())
            node_b = pydot.Node(b,label=k, style="filled", fillcolor="green")
            draw(graph,node_a, node_b)
            count=b
            # drawing the label using a distinct id
            a = (count)
            node_a = pydot.Node(a,label=k, style="filled", fillcolor="red")
            graph.add_node(node_a)
            b= (next_id())
            node_b = pydot.Node(b,label=str(v), style="filled", fillcolor="green")
            graph.add_node(node_b)
            draw(graph,node_a, node_b)



def main():
    lines = []
    attributes = []
    words = []
    line =[]
    actual = []
    predicted = []
    leafValues = []
    wholeClassList = []
    wholeDataset = []
    global maxLevel
    maxLevel = 0
    train = open(r'C:\Prasanna\Spring13\ML\HW1\data\train.txt')
    test = open(r'C:\Prasanna\Spring13\ML\HW1\data\test.txt')
    # Formating the train and text documents in the form of lists for processing in the algorithm
    wordsTrain,attributesTrain = formatText(train)
    wordsTest,attributesTest = formatText(test)
    print 'Training Instances size:'+ str(len(wordsTrain))
    print 'Attributes:'+str(len(attributesTrain))
    for attr in attributesTrain:
        print attr
    print 'Testing Instances size:'+ str(len(wordsTest))
    # Sending second copy of training data as feature values need to be iterated on whole training dataset and not the reduced dataset which would miss out some values in the decision tree.
    counts = []
    tree = createTree(wordsTrain,attributesTrain,wholeClassList,wordsTrain,0,0)
    graph = pydot.Dot(graph_type='graph')
    visit(graph,tree,0,0)
    graph.write_png('ProbabilisticTree.png')
    print 'maxLevel='+str(maxLevel)
    for i in range(0,maxLevel):
        tree=pruneUniformDistributions(tree,counts,[0],[0])
        print '\n'
        print tree
        tree=removeIndependentNodes(tree,counts,[0,0],[0,0],[0,0])
        print '\n'
        print tree
        counts=[]
    graph = pydot.Dot(graph_type='graph')
    visit(graph,tree,0,0)
    graph.write_png('PrunedProbabilisticTree.png')


if __name__ == '__main__':
    main()
