# markdown## About Me



I'm Sahaj, a 14-year-old self-taught  Aspiring ML Researcher  based in Bengaluru, India.



My interest in science started early — I was obsessed with space, black holes, 

and spacetime as a kid. That curiosity naturally pulled me toward quantum physics, 

and eventually toward quantum computing. When I discovered that machine learning 

and quantum computing were converging into an entirely new field — Quantum Machine 

Learning — I knew that was where I wanted to be.



I've spent the last 3-4 years building theoretical foundations across mathematics, 

physics, and computer science .But there was a problem I was always scared to take to into practice but  this portfolio represents the shift from theory to 

practice — moving from understanding concepts to actually building things.



My goal is to pursue a BSc in Computer Science at TU Delft with a Quantum Computing 

minor, followed by an MSc in AI, working towards research at the intersection of 

machine learning and quantum systems.



\## Technical Skills

\- \*\*Languages\*\* — Python

\- \*\*ML Libraries\*\* — Scikit-learn, SciPy

\- \*\*Data Processing\*\* — NumPy, Pandas

\- \*\*Visualization\*\* — Matplotlib

\- \*\*Version Control\*\* — Git, GitHub

\## Algorithms



\### Linear Regression

The first algorithm I learned — and honestly a great starting point. 

It finds the relationship between input features and a continuous output 

by fitting the best possible straight line through the data. Under the hood 

it minimizes the gap between predicted and actual values using Ordinary Least 

Squares. Simple, interpretable, and still widely used in practice.



\### K-Nearest Neighbours (KNN)

A deceptively simple algorithm — to classify a new point, just look at 

its K nearest neighbors and go with the majority vote. No training phase, 

just pure memory. It works surprisingly well on small datasets but struggles 

as data grows since it stores everything and searches through all of it 

at prediction time.



\### Support Vector Machine (SVM)

This one took the most effort to understand properly — and it was worth it. 

SVM finds the hyperplane that separates two classes with the maximum possible 

margin. The points sitting right on the boundary are called support vectors 

and they're the only ones that actually matter. What makes SVM powerful is 

the kernel trick — it implicitly maps data into higher dimensions where 

non-linear problems become linearly separable. I implemented this from scratch 

before using sklearn, which gave me a much deeper understanding of the 

optimization problem underneath.



\### XGBoost (Extreme Gradient Boosting)

Where KNN memorizes and SVM finds boundaries, XGBoost learns from mistakes. 

It builds decision trees sequentially — each tree focuses on correcting 

what the previous one got wrong. The result is an incredibly accurate model 

that's also fast and resistant to overfitting. It's the go-to algorithm 

for structured data and dominates competitive ML for good reason.



\### Random Forest Regression

Similar idea to XGBoost but the trees work independently rather than 

sequentially. Each tree is trained on a random subset of data and the 

final prediction is the average across all trees. The randomness is actually 

a feature — it prevents any single tree from overfitting and makes the 

overall model much more robust.



\### K-Means Clustering

My introduction to unsupervised learning. Without any labels, K-Means 

partitions data into K groups by minimizing the distance between points 

and their cluster centers. You specify K upfront — I used the Elbow Method 

to find the right number — and the algorithm iteratively refines the clusters 

until they stabilize. Applied this to both Titanic passenger data and 

mall customer segmentation.



\### Mean Shift Clustering

What I like about Mean Shift is that you don't need to tell it how many 

clusters to find — it figures that out itself. It works by sliding a window 

toward the densest regions of data using gradient ascent until it converges 

on natural peaks. More flexible than K-Means for exploratory analysis where 

you genuinely don't know the structure of your data upfront.



\### Hierarchical Clustering (Agglomerative)

The most visually informative clustering algorithm I've worked with. 

It builds a dendrogram — a tree showing how every point relates to every 

other point and at what level they merge. You don't commit to a fixed number 

of clusters upfront — you look at the tree and cut it where the gaps are 

largest. On the Titanic dataset it cleanly separated passengers into a 

26% survival group and an 80% survival group, which maps almost perfectly 

to the historical "women and children first" pattern.



\### Principal Component Analysis (PCA)

Less an algorithm, more a tool I reach for whenever data gets unwieldy. 

PCA finds the directions in your data where variance is highest — called 

principal components — and projects everything onto those axes. The result 

is fewer dimensions with most of the important information preserved. 

I used it alongside K-Means for customer segmentation, reducing features 

down to 2 dimensions so the clusters could actually be visualized.



\## Projects



\### Supervised Learning



| Project | Algorithm | Dataset | Accuracy |

|---|---|---|---|

| House Price Prediction | Linear Regression | Custom Dataset | — |

| Cancer Classification | SVM — RBF Kernel | Breast Cancer Wisconsin | 98.24% |

| Handwritten Digit Recognition | SVM — RBF Kernel | MNIST (70,000 images) | 96.15% |

| Stock Price Prediction | Linear Regression | Google Stock 2016 | — |



\### Unsupervised Learning



| Project | Algorithm | Dataset | Key Finding |

|---|---|---|---|

| Titanic Survival Clustering | K-Means | Titanic Passenger Data | 73% survival alignment |

| Customer Segmentation | K-Means + PCA | Mall Customers | 3 distinct segments |

| Titanic Survival Analysis | Mean Shift | Titanic Passenger Data | 5 natural passenger groups |

| Titanic Survival Analysis | Hierarchical Clustering | Titanic Passenger Data | 26% vs 80% survival split |



\---



\## Roadmap



\- \[x] Supervised Learning — Regression, Classification

\- \[x] Unsupervised Learning — Clustering, Dimensionality Reduction

\- \[ ] Neural Networks — PyTorch

\- \[ ] Deep Learning — CNNs, RNNs, Transformers

\- \[ ] Reinforcement Learning

\- \[ ] Quantum Machine Learning — PennyLane, Qiskit



\## Note

I made a point of understanding the math before touching sklearn. 

For SVM specifically, I built the full optimization loop from scratch — 

the margin maximization, constraint checking, and hyperplane visualization — 

before ever calling SVC(). I think that's the right way to learn this stuff. 

Libraries are tools, not shortcuts.



