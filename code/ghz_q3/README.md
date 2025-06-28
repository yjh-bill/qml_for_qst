## QST for 3-qubit GHZ state

#### Note

By default, for each iteration, SPSA algorithm evaluates $f(\vec{x}+\vec{\Delta})$ and $f(\vec{x}-\vec{\Delta})$, making two function calls. However, if one would like to know how the training loss evolves for $f(\vec{x})$, then an additional call is needed. Hence, there are two versions of code. To obtain the data for loss-vs-iteration, use `Alt_ideal_ghz_q3_b27_d10_sh100_eps3_plotloss.ipynb`. To save computation resource, use `Alt_ideal_ghz_q3_b27_d10_sh100_eps3_trial0_visual.ipynb`.
