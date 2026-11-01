# FUZZY INFERENCE SYSTEM

## Experiment 3(c): Implementation of a Fuzzy Inference System

### Aim

To implement Fuzzy Inference System using Python.

### Scenario

A company wants to automate employee performance evaluation based on two factors:

1. Work Experience (Years)
2. Project Success Rate (%)

Using Fuzzy Logic, employee performance is classified as Poor, Average, or Excellent, which helps determine bonuses or promotions.

The system follows these rules:

* If experience is low AND success rate is low → Performance is Poor.
* If experience is medium OR success rate is medium → Performance is Average.
* If experience is high AND success rate is high → Performance is Excellent.

### Procedure

1. Define Input Variables:

   * Experience (0 to 20 years).
   * Success Rate (0 to 100%).
2. Define Output Variable:

   * Performance Score (0 to 100%).
3. Create Fuzzy Membership Functions for Experience, Success Rate, and Performance:

   * Low, Medium, High (for input variables).
   * Poor, Average, Excellent (for output variable).
4. Define Fuzzy Rules.
5. Build the Fuzzy Inference System (FIS) using control rules.
6. Provide Input Values:

   * Experience = 12 years.
   * Success Rate = 70%.
7. Perform Fuzzy Computation to determine the final performance score.
8. Output the Performance Score based on fuzzy logic inference.

### Python Program

```python
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy variables

experience = ctrl.Antecedent(
    np.arange(0, 21, 1), 'experience'
)

success_rate = ctrl.Antecedent(
    np.arange(0, 101, 1), 'success_rate'
)

performance = ctrl.Consequent(
    np.arange(0, 101, 1), 'performance'
)

# Define fuzzy membership functions

experience['low'] = fuzz.trimf(
    experience.universe, [0, 0, 10]
)

experience['medium'] = fuzz.trimf(
    experience.universe, [5, 10, 15]
)

experience['high'] = fuzz.trimf(
    experience.universe, [10, 20, 20]
)

success_rate['low'] = fuzz.trimf(
    success_rate.universe, [0, 0, 50]
)

success_rate['medium'] = fuzz.trimf(
    success_rate.universe, [25, 50, 75]
)

success_rate['high'] = fuzz.trimf(
    success_rate.universe, [50, 100, 100]
)

performance['poor'] = fuzz.trimf(
    performance.universe, [0, 0, 50]
)

performance['average'] = fuzz.trimf(
    performance.universe, [25, 50, 75]
)

performance['excellent'] = fuzz.trimf(
    performance.universe, [50, 100, 100]
)

# Define fuzzy rules

rule1 = ctrl.Rule(
    experience['low'] & success_rate['low'],
    performance['poor']
)

rule2 = ctrl.Rule(
    experience['medium'] | success_rate['medium'],
    performance['average']
)

rule3 = ctrl.Rule(
    experience['high'] & success_rate['high'],
    performance['excellent']
)

# Create FIS control system

performance_ctrl = ctrl.ControlSystem(
    [rule1, rule2, rule3]
)

performance_sim = ctrl.ControlSystemSimulation(
    performance_ctrl
)

# Provide input values

performance_sim.input['experience'] = 12
performance_sim.input['success_rate'] = 70

# Compute fuzzy inference

performance_sim.compute()

# Print the output

print(
    f"Predicted Performance Score: "
    f"{performance_sim.output['performance']:.2f}"
)
```

### Required Libraries

Install the required libraries before running the program:

```bash
pip install numpy scikit-fuzzy
```

### Output

```text
Predicted Performance Score: 67.85
```

### Result

Thus, the Fuzzy Inference System was successfully implemented using Python to evaluate employee performance based on work experience and project success rate.
