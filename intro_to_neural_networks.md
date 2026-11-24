# Intro to Neural Networks: Course Reading Summary

Source: https://scaler-content.github.io/DSML-NN-1/

> Note: this is a summary of the page, not a verbatim copy.

## Running case

Hexa Cell, a lithium-ion cell plant, must sort about 40,000 cells a day into three grades:

- **A:** ship
- **B:** rework
- **C:** scrap

The reading uses 1,200 logged cells with two features: resistance deviation (x1) and temperature-rise deviation (x2).

## 1. Why a straight line fails

- Healthy cells sit near nominal values, and faulty cells drift far in either direction. The classes form **rings**, which a straight boundary cannot separate.
- Feature engineering can change the problem. Adding r² = x1² + x2² raised logistic regression from about 0.34 to about 0.97 accuracy without changing the model.

## 2. Diagnosing the existing toolkit

| Model | Limitation |
|-------|------------|
| Logistic regression | Linear boundaries only; multi-class needs wrappers |
| K-nearest neighbours | Handles curves, but stores all training data and slows as data grows |
| Decision trees | Approximate curves with axis-aligned staircases; poor on high-dimensional input like pixels |
| Kernel SVM | Handles curves with an RBF kernel, but needs an m × m kernel matrix that becomes unaffordable at scale |

Quiz: why does a tree fail on 65,538 pixel features? Each split tests one feature, so it cannot capture patterns spread across pixels.

## 3. Two jobs in classical ML

- **Job 1:** inventing a good representation. Humans do this by hand.
- **Job 2:** drawing a boundary in that representation. Classical algorithms do this well.

Three requirements follow: learned features, native multi-class support, and fixed inference cost.

## 4. Everyday examples

Ad ranking, autoencoders for compression, image segmentation with inpainting, and text prediction. Each learns its own intermediate representation from raw input.

## 5. From biology to the neuron

- A supervisor in a control room combines weighted reports and acts past a threshold. This maps to a neuron: features, weights, weighted sum, bias, activation, output.
- Weights are signed importances, the bias belongs to the unit, and both are learned.
- Quiz lesson: raw weight magnitudes are only comparable after features are standardised.

## 6. The mathematics

- Net input: `z = wᵀx + b`. The sign of z gives the side of a hyperplane, and `|z| / ‖w‖` gives the distance from it.
- Activations: step, sigmoid, tanh, ReLU. The activation determines the output shape, not the location of the boundary.

## 7. Why non-linearity is essential

Stacking linear operations only produces another linear function. Activations let layers build more complex representations.

## 8. Matrix form and playgrounds

- Batch computation: `Z = XW + b`, `A = f(Z)`, with shapes (m, d), (d, h) and (m, h).
- A single-neuron playground shows that no setting of its parameters encloses the ring-shaped healthy region.

## 9. Known models as single neurons

- Logistic regression = neuron with sigmoid.
- Perceptron = neuron with step (1957).
- Linear SVM = neuron with hinge-based loss and L2 penalty.

## 10. Multi-class via softmax

One output unit per class, then softmax normalises scores into probabilities that sum to one. With two classes, it reduces to sigmoid.

## 11. Networks

- Many hidden units act as many straight cuts that together enclose curved regions.
- Parameters per layer: `n_in × n_out + n_out`. The example network (2 → 5 → 3) has 33 parameters.
- Quiz lesson: a deep network without activations collapses to a single linear map and cannot learn the rings.
- Capacity playground: four ReLU hidden units (27 parameters) reach about 0.97 test accuracy.

## 12-13. History and data volume

- The mathematics existed by 1986. Data and compute limited adoption until later.
- Classical methods perform well with little data and plateau. Neural networks start slower but keep improving with more data. Data volume is a major factor in choosing a model family.

## 14. AI study exercises

- Trace the lineage from one neuron to modern language models, naming the limitation each step removes.
- Check a worked forward-pass example containing one planted arithmetic error.
- Choose a classifier for your own data, naming a specific mechanism for each failure.

## 15. Wrap-up

The session covered the limits of the existing toolkit, the neuron as weighted sum plus activation, the network as stacked neurons with softmax outputs, and backpropagation as the topic of the next session.
