# CPS 491 - Capstone II

Source: <https://github.com/cps491sp21-team13/cps491sp21-main-repo.git>

University of Dayton

Department of Computer Science

CPS 491 - Capstone II, Semester Year

Instructor: Dr. Phu Phung


## Capstone II Project 


# Deriving Multi-layer Scaffolding of Compositional Neural Networks from existing, monolithic networks

# Team members

1.  Zachary Rowland, rowlandz1@udayton.edu


# Company Mentor

Matthew Clark, Principal Scientist

Galois

444 E 2nd Street

Dayton, OH 45402


# Project Management Information

Management board (private access): <https://trello.com/b/kVF2rdUV/cps-491-team-13>

Source code repository (private access): <https://github.com/cps491sp21-team13/cps491sp21-main-repo>

Project homepage (public): <https://cps491sp21-team13.github.io/>

## Revision History

| Date       |   Version     |  Description |
|------------|:-------------:|-------------:|
| 16/02/2021 |  0.1          | Details of Taylor and convolutional approximations |
| 16/03/2021 |  0.2          | Details data-driven SINDy approach/change-of-basis strategy for building neural networks |
| 05/04/2021 |  0.3          | Preliminary results of applying regression-based model fitting to neural networks |
| 04/05/2021 |  1.0          | Evolutionary algorithm implemented |


# Overview

The goal of this project is to explore and document the correlation between the configuration of deep neural
networks and the analytical mathematical functions that they model. A long term goal, possibly not explored
in this project specifically, is to develop methods for _decomposing_ an arbitrary deep neural network into
smaller networks with intuitive behavior, and _simplifying_ the network into the minimum network needed to
model the desired mathematical function.

To this end, I have identified a method of reducing a neural network to a smaller mathematical model using regression-strategies. To test the capabilities of this reduction method, I build an evolutionary algorithm that learns a concise polynomial approximation of a neural network trained to classify handwritten digits.

# Project Context and Scope

This project is a contribution to research into the development of machine learning models that learn intuitively.

## High-level Requirements

By the end of the semester, we hope to have reasonably comprehensive documentation of the different ways simple neural networks can approximate simple functions. The process of compiling this documentation should involve careful inspection of trained neural networks guided by mathematics.

# Implementation

Most work on this project is being done in a Jupyter notebook using markdown for mathematics
and Tensorflow for implementing neural networks. However, work migrated to a LaTeX file as the work started to become finalized.

The core of the evolutionary algorithm is given below. The algorithm performs a cycle of mutation and selection on a bank of candidate equations. Individual pixels of the input image are slowly added to the equations over time in order of decreasing variance across the image data set. The equations are mutated according to a metric on terms called VIC (Variance times Inverse Coefficient) in an attempt to speed up the learning process. This algorithm is explained further in the LaTeX document.

    n = 10         # number of equations to work with at once
    numterms = 10  # number of terms in each equation
    
    x_data, y_data = getDataForDigit(0, model=model)
    
    # Gather pixels with a variance above the threshold 0.1
    sigterms = Equation()
    for i, singlepixeldata in enumerate(x_data.T):
      sigterms._terms.append(Term("x_{%d}" % i, singlepixeldata, 1))
    sigterms.elimByVarThreshold(0.1)
    sigterms.sortByVariance()
    
    # Build initial linear equations from highest-varying pixels
    eqs = []
    for i in range(n):
      neweq = Equation(sigterms._terms[:numterms])
      neweq.fitTerms(y_data)
      eqs.append(neweq)
    
    # Evolutionary learning loop
    lintermix = numterms    # next linear term to add
    iterations = 0          # number of iterations
    while lintermix < len(sigterms._terms):
      iterations += 1
    
      # Mutation: Add the next highest-varying linear term
      for eq in eqs:
        eq._terms.append(sigterms._terms[lintermix])
        eq.fitTerms(y_data)
        eq.pruneByCoef(numterms)
        eq.fitTerms(y_data)
      lintermix += 1
    
      # Mutation: Recombination of a single low-VIC term
      for eq in tuple(eqs):
        neweq = eq.mutate()
        neweq.fitTerms(y_data)
        neweq.pruneByCoef(numterms)
        neweq.fitTerms(y_data)
        eqs.append(neweq)
    
      # Selection: remove equations with low R^2 value
      eqs.sort(key=Equation.getR2, reverse=True)
      eqs = eqs[0:n]
    
      # Evaluation & Display
      averageR2 = sum(map(Equation.getR2, eqs)) / len(eqs)
      if iterations % 10 == 0 or iterations < 10:
        print("Iterations: {}, R^2: {}".format(iterations, averageR2))
    
    print("----- Final equation -------")
    eqs.sort(key=Equation.getR2, reverse=True)
    eqs[0].display()
    print("R^2:", eqs[0].getR2())

# Software Process Management

## Scrum process

### Sprint 0

Duration: 20/01/2021-27/01/2021

#### Completed Tasks: 

1. Studied fuzzy inference systems from material provided from Galois
2. Studied automatic differentiation.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| learned a lot of preliminary information | documentation of progress | take more notes on what I read about/discover  |

### Sprint 1

Duration: 27/01/2021-03/02/2021

#### Completed Tasks:

1. Learned tensor manipulation and learning models in Tensorflow
2. Experimented with small neural networks in a Jupyter notebook

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| good pace aquiring new knowlege and documenting what I have learned | could have started experimenting with small networks sooner | reach out to ask questions sooner |

### Sprint 2

Duration: 03/02/2021-09/02/2021

#### Completed Tasks:

1. Learned about convolution from Linear Systems.
2. Detailed a technique for constructing neural networks that approximate arbitrary
   functions inspired by convolution.
3. Detailed a technique for approximating arbitrary polynomials with neural networks
   based on Taylor-series expansion.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| documentation was thorough and clear | direction of work could have been more focused toward project goal | more formal communication about the precise goal of the project |

### Sprint 3

Duration: 28/02/2021-09/02/2021

#### Completed Tasks:

1. Explored Taylor series approximations and detailed a method of
   building a neural network using a Taylor series approximation
2. Explored Fourier series approximations

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Documentation was clear | Better communication with company sponsor regarding the direction of the project | smaller, more frequent communication |

### Sprint 4

Duration: 01/03/2021-16/03/2021

#### Completed Tasks:

1. Learned about spline-theory as it relates to deep learning.
2. Learned about SINDy approach to fitting functions to data.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Learning was done quickly and efficiently | Could have implemented ideas more instead of just detailing them in math | organize notebook and code |

### Sprint 5

Duration: 16/03/2021-24/03/2021

#### Completed Tasks:

1. Applied SINDy to a network trained on the MNIST dataset.
2. Built Python infrastructure for manipulating equations and data as required by SINDy.
3. Compared linear and quadratic models learned from SINDy for classifying the a digit.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Very productive in terms of documentation and implementation. Concrete results were found. | NA | Continue with this line of work. |

### Sprint 6

Duration: 24/03/2021-05/04/2021

#### Completed Tasks:

1. Wrote project introduction and motivation.
2. Literature review on approximation theory of neural networks and model compression.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Productivity | NA | Continue literature review |

### Sprint 7

Duration: 05/04/2021-19/04/2021

#### Completed Tasks:

1. Built evolutionary equation-learning algorithm and tested it on an MNIST network
2. Documented observasions about the evolutionary algorithm and one of the equations it learned.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Lots of practical code written | Documentation, notebook is a little messy | spend more time tidying notebook |

### Sprint 8

Duration: 19/04/2021-04/05/2021

#### Completed Tasks:

1. Ran experiments testing viability of the VIC metric and high-variance term adding.
2. More documentation and writing.

#### Sprint Retrospection:

| Good     |   Could have been better    |  How to improve?  |
|----------|:---------------------------:|------------------:|
| Experiments | Writing is slow | figure out how to write more efficiently |

# Acknowledgments 

I would like to thank Mr. Matthew Clark, Principal Scientist at Galois, for mentoring this project.
