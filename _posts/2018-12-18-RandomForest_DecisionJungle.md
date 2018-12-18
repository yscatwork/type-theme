---
layout: post
title: Random Forest & Decision Jungle
description: "Random Forest & Decision Jungle"
headline: "Random Forest & Decision Jungle"
categories: MACHINELEARNING
tags: 
  - Random Forest
  - Decision Jungle
comments: false
mathjax: true
featured: true
published: true
---

# 0. 앙상블, 그리고 배깅 (Ensemble & Bagging)

앙상블 기법(Ensemble method)이란, 여러개의 예측기 (classifier/regressor)로부터 얻은 출력 결과를 적절히 결합하여 보다 정확한 예측을 도출하는 방법입니다. 

앙상블 기법의 하나인 배깅(Bagging)은 Bootstrap Aggregating의 뜻으로, 전체 데이터로부터 무작위 복원 방식으로 subset(=bootstrap) 데이터를 여러개 추출하여 이로부터 여러개의 예측기를 훈련하는 방식입니다. 

같은 입력 값에 대해 예측기들이 내놓는 서로 다른 출력 값들은 majority/weighted voting 또는 stacking 등의 여러 방식을 통해 합산(aggregate)됩니다. 

예측기는 어떤 지도학습 알고리즘을 사용해서 만들어도 됩니다.


# 1. 결정 트리와 랜덤 포레스트 (Random Forest):

그 중에서도 배깅의 **예측기가 결정 트리**일 때를 **랜덤 포레스트**라 합니다. 

결정 트리는 일련의 질문들을 통해 데이터를 분류하는 방법입니다. 

결정 트리는 과적합에 취약하지만, 배깅 방식으로 여러 트리를 결합하면 이런 한계를 극복할 수 있습니다.


랜덤 포레스트는 부트스트랩 샘플링과 변수 무작위 추출의 두 가지 방법으로 앙상블의 다양성을 확보합니다. 


<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/2.png" width="500"></p>


랜덤 포레스트는 트리 pruning을 하지 않습니다. 따라서 훈련 데이터에 과적합할 수 있는 위험이 있습니다. 랜덤 포레스트의 Generalization Error 는 각 트리간의 모델 독립성 (p bar)이 낮을 수록, 그리고 개별 트리의 정확도가 (s^2) 높을 수록 낮아집니다. 

<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/3.png" width="200"></p>

## 1.1 변수별 중요도 (Variable Importance)

Bootstrapping 시에 선택 되지 않은 데이터를 Out-of-Bag (OOB)데이터라 합니다. 

OOB 데이터를 분류기에 입력으로 넣어 inference를 할때, 만약 그대로 넣는 것과 변수별 값을 뒤섞어 넣는 것의 출력 차이가 크지 않다는 것은 무슨 뜻일까요? 

값이 뒤바뀌어도 크게 상관이 없다, 즉 값이 뒤섞인 변수가 결정 트리에서 별로 중요하지 않다는 뜻입니다. 

이런 논리로 OOB 데이터로부터 변수별 중요도를 계산 할수 있습니다.
<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/4.png" width="500"></p>

## 1.2 코드 예시
참고 코드: https://github.com/llSourcell/random_forests


우선 전체 데이터로부터 일부 데이터를 bootstrap하는 함수를 정의합니다. 


*인풋: [전체 데이터, bootstrap 비율]*

```python
def bootstrap(alldata,ratio):
	samples = []
	n_sample = round(len(dataset)*ratio))
	while len(samples) < n_sample:
		index = randrange(len(alldata))
		samples.append(alldata[index])
	return samples
```

위에서 bootstrap된 데이터로 트리를 만드는 과정을 둘로 나눕니다. 

(1) 가장 좋은 분기점을 찾습니다.

(2) 해당 기준으로 recursive하게 트리를 split 합니다.


기본적으로 결정 트리 훈련은 데이터를 잘 양분하기 위한 일련의 기준들을 찾아나가는 과정입니다. 

좋은 기준이란 무엇일까요? 단순히 생각하면, 특정 부모 노드에서 두 자녀 노드로 5:5로 양분되는 것은 좋은 기준이라고 할 수 없습니다 (인퍼런스 데이터가 들어왔을때 왼쪽으로 가도 그만, 오른쪽으로 가도 그만이기 때문입니다). 

이보다는 한 쪽으로 확실히 치우친 분류가 가능할 수록 효과적인 분류 기준입니다. 이를 수치화 하는 것이 아래의 GINI 계수입니다.

Gini(t)=1-\sum_{0}^{c-1}[p(i|t)]^2

```python
def gini_idx(groups, class_values):
	gini_idx = 0.0
   #각 class 노드에 대하여 아래를 계산합니다
	for class_value in class_values:
		for group in groups:
			size = len(group)
			if size == 0:
				continue
        # 모든 클래스 값들의 평균을 계산.
			p = [row[-1] for row in group].count(class_value) / float(size)
        #  지니계수 공식에 따라 모든 (p * 1-p) 값들을 더해나감
			gini_idx += (proportion * (1.0 - proportion))
	return gini_idx
```

지니 계수를 최소화 하는 방향으로 진행되는 데이터 분기점을 찾는 과정은 다음과 같습니다. 

*인풋: [bootstrap, 사용할 피쳐수]*


```python
def get_best_split(bootstrap,num_feats): 
	b_index,b_value,b_score,b_groups = 999,999,999, None
	features = []
	 
	# 지정해준 숫자만큼의 feature를 랜덤하게 선택합니다.
	while(len(features) < num_feats)
		index = ranrange(len(bootstrap[0])-1)
		if index not in features:
			features.append(index)
			
	# 피쳐 선택시 마다의 gini index를 구합니다.
	for index in features:
		for row in bootstrap:
			groups = test_split(index,row[index],bootstrap)
			gini_idx = gini_index(groups, class_values)
			if gini_idx < b_score: #업데이트합니다
				b_index,b_value,b_score,b_groups = index,row[index],gini,groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
```

위와 같이 찾아낸 좋은 root node 로 tree splitting을 진행합니다. 

아래 코드는 트리의 최대 depth, child node 최소 크기 조건이 만족되면 terminal node를 생성합니다. 

그 전 까지는 root note 선정을 반복하고 split 함수 스스로를 recursive 하게 실행합니다.

*인풋: [rootnode, 트리 최대 깊이, child 노드 최소크기, 사용할 피쳐수, 최초depth]*
    
```python
def split(rootnode, mx_depth, mn_size, num_feats, depth):
	left, right = node['groups']
	del(node['groups'])
	
   # 만약 left 나 right이 비어있다면(=더이상 split없음) terminal 노드를 생성합니다.
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
		
   # max depth에 도달했다면 terminal 노드를 생성합니다.
   if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
		
   # right & left child가 각각 min_size 이하이면 terminal node를 생성합니다. 
   # min_size보다 크다면 depth-first 방식으로 recursive하게 split을 진행합니다. 
    
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_best_split(left, num_feats)
		split(node['left'], mx_depth, mn_size, num_feats, depth+1)
		
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_best_split(right, num_feats)
		split(node['right'], mx_depth, min_size, num_feats, depth+1)
		
```

만들어진 트리를 이용해 test 데이터에 대한 아웃풋을 출력하는 방식 또한 recursive하게 정의합니다.

```python

def predict(node, row):
    #called again with the left or the right child nodes, depending on how the split affects the provided data. Check if a child node is either a terminal value to be returned as the prediction or if it is a dictionary node containing another level of the tree to be considered.
    
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
```

가장 상위인 main 함수인 random_forest는 training 데이터와 파라미터로 랜덤포레스트를 훈련하고 
test 데이터를 이용한 인퍼런스 결과를 내놓습니다. 

*인풋: [훈련 데이터, 테스트 데이터, 트리 최대 깊이, child 노드 최소크기, bootstrap 사이즈, 총 트리개수, 사용할 피쳐수]* 
		
```python

def bagging_result(alltrees, row):
	predictions = [predict(tree, row) for tree in alltrees]
	return max(set(predictions), key=predictions.count)

def randomforest(traindata, testdata, mx_depth, mn_size, sample_size, num_trees, num_feats):
	trees = []
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, mx_depth, mn_size, num_feats)
		trees.append(tree)
		
	#test 데이터 엔트리마다 random forest 앙상블 모델을 이용한 예측을 합니다.
	predictions = [bagging_result(trees, row) for row in test] 
	
	return(predictions)
	
```

데이터 위치: [sonar.all-data.csv](https://github.com/yscatwork/yscatwork.github.io/blob/branch/sonar.all-data.csv)

코드를 모두 실행하려면 아래의 코드 블락을 실행합니다.

```python
from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt

# 데이터 가져오기
def get_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		c = reader(file)
		for row in c:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = [], []
    #for every row
	for row in dataset:
        #if the value at that row is less than the given value
		if row[index] < value:
            #add it to list 1
			left.append(row)
		else:
            #else add it list 2 
			right.append(row)
    #return both lists
	return left, right
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    #how many correct predictions?
	correct = 0
    #for each actual label
	for i in range(len(actual)):
        #if actual matches predicted label
		if actual[i] == predicted[i]:
            #add 1 to the correct iterator
			correct += 1
    #return percentage of predictions that were correct
	return correct / float(len(actual)) * 100.0
 
def gini_index(groups, class_values):
	gini = 0.0
    #for each class
	for class_value in class_values:
        #a random subset of that class
		for group in groups:
			size = len(group)
			if size == 0:
				continue
            #average of all class values
			proportion = [row[-1] for row in group].count(class_value) / float(size)
            #  sum all (p * 1-p) values, this is gini index
			gini += (proportion * (1.0 - proportion))
	return gini
 
def get_best_split(bootstrap, num_feats):
	class_values = list(set(row[-1] for row in bootstrap))
	b_index, b_value, b_score, b_groups = 1000, 1000, 1000, None #큰 숫자로 최소화 해둡니다.
	features = list()
	while len(features) < num_feats:
		index = randrange(len(bootstrap[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in bootstrap:
			groups = test_split(index, row[index], bootstrap)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)
 
def split(rootnode, mx_depth, mn_size, num_feats, depth):
	left, right = rootnode['groups']
	del(rootnode['groups'])
    
	if not left or not right:
		rootnode['left'] = rootnode['right'] = to_terminal(left + right)
		return
	
	if depth >= mx_depth:
		rootnode['left'], rootnode['right'] = to_terminal(left), to_terminal(right)
		return

	if len(left) <= mn_size:
		rootnode['left'] = to_terminal(left)
	else:
		rootnode['left'] = get_best_split(left, num_feats)
		split(rootnode['left'], mx_depth, mn_size, num_feats, depth+1)

	if len(right) <= mn_size:
		rootnode['right'] = to_terminal(right)
	else:
		rootnode['right'] = get_best_split(right, num_feats)
		split(rootnode['right'], mx_depth, mn_size, num_feats, depth+1)
 
def build_tree(train, mx_depth, mn_size, num_feats):
	root = get_best_split(train, num_feats)
	split(root, mx_depth, mn_size, num_feats, 1)
	return root
 
def getpredict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return getpredict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return getpredict(node['right'], row)
		else:
			return node['right']
 
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

def bagging_do_predict(trees, row):
	predictions = [getpredict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)
 

def randomforest(traindata, testdata, mx_depth, mn_size, sample_size, num_trees, num_feats):
	trees = []
	for i in range(num_trees):
		sample = subsample(traindata, sample_size)
		tree = build_tree(sample, mx_depth, mn_size, num_feats)
		trees.append(tree)
	
	predicts = [bagging_do_predict(trees, row) for row in testdata]
	
	return(predicts)


#################### RUN #########################	
# load and pre-process data (자료형변경)
dataset = get_csv('sonar.all-data.csv')
for i in range(0, len(dataset[0])-1):
	for row in dataset:
		row[i] = float(row[i].strip())

# M,R로 된 label을 0,1로 수정
col=len(dataset[0])-1
class_values = [row[col] for row in dataset]
unique = set(class_values)
lookup = dict()
for i, value in enumerate(unique):
    lookup[value] = i
for row in dataset:
    row[col] = lookup[row[col]]

#shuffle and split test & train data
total_len = len(dataset)
index = list(range(0, total_len))
shuffle(index)
test_ratio=0.1
train_idx=int(total_len*(1-test_ratio))
traindata=dataset[:train_idx]
testdata=dataset[train_idx:]

# randomforest 실행
mx_depth = 10
mn_size = 1
sample_size = 1.0
num_feats = int(sqrt(len(dataset[0])-1))
num_trees=5

predict = randomforest(traindata, testdata, mx_depth, mn_size, sample_size, num_trees, num_feats)
target = [ row[-1] for row in testdata ]
accuracy = accuracy_metric(target, predict)
print("   target labels: {}".format(target))
print("predicted labels: {} ".format(predict))
print("accuracy: {} %".format(accuracy))

```

# 2. Decision Jungle

랜덤 포레스트는 높은 성능으로 여러 분야에서 널리 사용되고 있습니다.

그러나 알고리즘의 특성상 depth와 함께 노드 width 증가 또한 폭발적이기 때문에 메모리 부담이 커지게됩니다.

Decision Jungle은 이를 타개하기 위해 tree 그래프 대신에 Directed Acyclic Graph (DAG) 로의 변형을 꾀합니다. 

DAG에서는 아래의 그림에서 처럼, child node가 여러개의 부모로부터 올 수 있습니다. 

엣지(edge)와 노드(node)의 효율적인 구성으로 인해 분류기의 depth 증가에 따른 width 증가가 크게 줄어듭니다. 

이러한 DAG를 만들기 위해서, rotation forest와는 달리 주기적으로 node들을 합치는 과정이 추가됩니다.

<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/6.png" width="500"></p>


## 2.1 구성
Decision Jungle은 아래처럼 정리할 수 있습니다.

<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/7.png" width="300"></p>

Np: 부모 노드

Nc: 자식 노드 (파라미터로 주어진다)

θi: 부모 레벨 노드 i에서의 split feature function

Si: 노드 i로 부터 왼쪽 또는 오른쪽으로 분류되는 데이터

li in Nc: parent node i 에서 왼쪽 edge를 타고 온 결과물

ri in Nc: parent node i 에서 오른쪽 edge를 타고 온 결과물

Sj({θi},{li},{ri}): 자식 노드에 도착하는 결과물 (= 부모 노드의 왼쪽 엣지에서 온 것 + 부모 노드 오른쪽 엣지에서 온 것의 합집합)



## 2.2 훈련

Decision Jungle의 훈련시 목적함수는 아래와 같이 정의됩니다.

<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/8.png" width="200"></p>

Split feature function과, 자식 assginment 둘을 joint로 최소화 하는 것이 훈련의 방향입니다.

그러나 위의 목적식은 쉽게 풀기 어려우므로, 가장 그럴듯한 값으로 초기값 지정을 한 후에 L-search를 통해 최적값을 찾습니다.

<p align="center"><img src="https://github.com/yscatwork/yscatwork.github.io/blob/master/images/9.png" width="500"></p>


L-search는 split optimization과 branch optimization step이 번갈아 이루어지는 과정입니다. 

먼저, split optimization으로 부모 노드상에서 최적의 분기 값을 찾습니다. 

이때, 특정 자식 노드로 데이터의 집중이 높게 이루어 질때 엔트로피가 작아집니다.

다음에 이루어지는 branch optimization은 split 노드 값을 고정한 상태로 더이상의 entropy 변화가 없을때 까지 branching을 최적화합니다.


## 2.3 코드 예시

코드 https://github.com/gdanezis/trees/blob/master/code/forests.py

데이터: https://github.com/gdanezis/trees/edit/master/data/pg5711.txt
      https://github.com/gdanezis/trees/edit/master/data/pg23428.txt


#decision 정글을 생성하는 함수입니다.
**인풋 : 훈련데이터, 피쳐, 트리의 층위 레벨, 피쳐 개수**

```python
def build_jungle(train, features, levels=10, numfeatures=50):
    DAG = {0: copy.copy(train)}
    Candidate_sets = [0]
    next_ID = 0
    M = 20

    for level in range(levels):
        result_sets = []
        for tdata_idx in Candidate_sets:
            tdata = DAG[tdata_idx]

            if entropy(tdata) == 0.0:
                next_ID += 1
                idx1 = next_ID
                result_sets += [idx1]
                DAG[idx1] = tdata + []
                del DAG[tdata_idx][:]
                DAG[tdata_idx] += [True, idx1, idx1]
                continue

            X = (split(tdata, F) for F in random.sample(features, numfeatures))
            H, L1, L2, F = max(X)

            # Branch = (F, M1, M2)
            next_ID += 1
            idx1 = next_ID
            DAG[idx1] = L1
            next_ID += 1
            idx2 = next_ID
            DAG[idx2] = L2

            result_sets += [idx1, idx2]
            del DAG[tdata_idx][:]
            DAG[tdata_idx] += [F, idx1, idx2]

        ## Now optimize the result sets here
        random.shuffle(result_sets)

        basic = result_sets[:M]
        for r in result_sets[M:]:
            maxv = None
            maxi = None
            for b in basic:
                L = float(len(DAG[r] + DAG[b]))
                e1 = len(DAG[r]) * entropy(DAG[r])
                e2 = len(DAG[b]) * entropy(DAG[b])
                newe = L * entropy(DAG[r] + DAG[b])
                score = abs(e1 + e2 - newe)
                if maxv is None:
                    maxv = score
                    maxi = b
                    continue
                if score < maxv:
                    maxv = score
                    maxi = b
            DAG[maxi] += DAG[r]
            del DAG[r]
            DAG[r] = DAG[maxi]

        Candidate_sets = basic

    for tdata_idx in Candidate_sets:
        tdata = DAG[tdata_idx]
        C1 = Counter([b for _, b in tdata])
        del DAG[tdata_idx][:]
        DAG[tdata_idx] += [None, C1]

    return DAG

#만들어진 디시전 정글로 분류를 수행합니다.

def classify_jungle(DAG, item):
    branch = DAG[0]
    while branch[0] is not None:
        try:
            fet, L1, L2 = branch
            if fet == True or fet in item:
                branch = DAG[L1]
            else:
                branch = DAG[L2]
        except:
            print len(branch)
            raise
    return branch[1]

```

아래의 코드 블럭을 실행하면 decision jungle 분류기가 훈련 & 실행됩니다.

```python

import random
from collections import Counter
import numpy as np
import copy
from csv import reader


def get_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        c = reader(file)
        for row in c:
            if not row:
                continue
            dataset.append(row)
    return dataset

def split_data(data, label=0, length=50):
    'Take a large text and divide it into chunks'
    strings = [data[i:i+length] for i in range(0, len(data) - length, length)]
    random.shuffle(strings)
    strings = [(s, label) for s in strings]
    test = strings[:int(len(strings) * 10 / 100)]
    training = strings[int(len(strings) * 10 / 100):]
    return test, training


def entropy(data):
    'Computes the binary entropy of labelled data'
    v = Counter([b for _, b in data]).values()
    d = np.array(v) / float(sum(v))
    return - sum(d * np.log(d))


def split(train, feat):
    'Split data according to an infromation gain criterium'
    ## first compute the entropy
    Hx = entropy(train)
    if Hx < 0.000001:
        raise Exception("Entropy very low")
    L1 = []
    L2 = []
    for t in train:
        if feat in t[0]:
            L1 += [t]
        else:
            L2 += [t]

    E1 = entropy(L1)
    E2 = entropy(L2)
    L = float(len(train))

    H = Hx - E1 * len(L1)/L - E2 * len(L2)/L
    return H, L1, L2, feat

def build_jungle(train, features, levels=20, numfeatures=100):
    DAG = {0: copy.copy(train)}
    Candidate_sets = [0]
    next_ID = 0
    M = 20

    for level in range(levels):
        result_sets = []
        for tdata_idx in Candidate_sets:
            tdata = DAG[tdata_idx]

            if entropy(tdata) == 0.0:
                next_ID += 1
                idx1 = next_ID
                result_sets += [idx1]
                DAG[idx1] = tdata + []
                del DAG[tdata_idx][:]
                DAG[tdata_idx] += [True, idx1, idx1]
                continue

            X = (split(tdata, F) for F in random.sample(features, numfeatures))
            H, L1, L2, F = max(X)

            # Branch = (F, M1, M2)
            next_ID += 1
            idx1 = next_ID
            DAG[idx1] = L1
            next_ID += 1
            idx2 = next_ID
            DAG[idx2] = L2

            result_sets += [idx1, idx2]
            del DAG[tdata_idx][:]
            DAG[tdata_idx] += [F, idx1, idx2]

        ## Now optimize the result sets here
        random.shuffle(result_sets)

        basic = result_sets[:M]
        for r in result_sets[M:]:
            maxv = None
            maxi = None
            for b in basic:
                L = float(len(DAG[r] + DAG[b]))
                e1 = len(DAG[r]) * entropy(DAG[r])
                e2 = len(DAG[b]) * entropy(DAG[b])
                newe = L * entropy(DAG[r] + DAG[b])
                score = abs(e1 + e2 - newe)
                if maxv is None:
                    maxv = score
                    maxi = b
                    continue
                if score < maxv:
                    maxv = score
                    maxi = b
            DAG[maxi] += DAG[r]
            del DAG[r]
            DAG[r] = DAG[maxi]

        Candidate_sets = basic

    for tdata_idx in Candidate_sets:
        tdata = DAG[tdata_idx]
        C1 = Counter([b for _, b in tdata])
        del DAG[tdata_idx][:]
        DAG[tdata_idx] += [None, C1]

    return DAG


def classify_jungle(DAG, item):
    branch = DAG[0]
    while branch[0] is not None:
        try:
            fet, L1, L2 = branch
            if fet == True or fet in item:
                branch = DAG[L1]
            else:
                branch = DAG[L2]
        except:
            print(len(branch))
            raise
    return branch[1]


dataEN = get_csv("./pg5711.txt")
dataFR = get_csv("./pg23428.txt")

length = 100

testEN, trainEN = split_data(dataEN, label=0, length=length)
testFR, trainFR = split_data(dataFR, label=1, length=length)

train = trainEN + trainFR
random.shuffle(train)
test = testEN + testFR
random.shuffle(test)

sometrain = random.sample(train, 50)
features = set()
while len(features) < 70:
    fragment, _ = random.choice(sometrain)
    l = int(round(random.expovariate(0.20)))
    b = random.randint(0, max(0, length - l))
    feat = fragment[b:b+l]

    ## Test
    C = 0
    for st, _ in sometrain:
        if feat in st:
            C += 1

    f = float(C) / 1000
    if f > 0.01 and f < 0.99 and feat not in features:
        features.add(feat)

features = list(features)

manytrees = []
jungle = []
for i in range(5):
    print("Build tree %s" % i)
    size = len(train) / 3
    training_sample = random.sample(train, size)

    tree = build_jungle(training_sample, features, numfeatures=100)
    jungle += [tree]

    tree = build_tree(training_sample, features, numfeatures=100)
    manytrees += [tree]

testdata = test
results_tree = Counter()
results_jungle = Counter()
for item, cat in testdata:
    # Jungle
    c = Counter()
    for tree in jungle:
        c += classify_jungle(tree, item)
    res = (max(c, key=lambda x: c[x]), cat)
    results_jungle.update([res])

print("Results         Tree   Jungle")
print("True positives:  %4d    %4d" % (results_tree[(1, 1)], results_jungle[(1, 1)]))
print ("True negatives:  %4d    %4d" % (results_tree[(0, 0)], results_jungle[(0, 0)]))
print ("False positives: %4d    %4d" % (results_tree[(1, 0)], results_jungle[(1, 0)]))
print ("False negatives: %4d    %4d" % (results_tree[(0, 1)], results_jungle[(0, 1)]))
```


20181216 조영선
