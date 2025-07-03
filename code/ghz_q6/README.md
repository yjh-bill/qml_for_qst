## QST for 6-qubit GHZ state

#### Note

By default, for each iteration, SPSA algorithm evaluates $f\left(\vec{x}+\vec{\Delta}\right)$ and $f\left(\vec{x}-\vec{\Delta}\right)$, making two function calls. However, if one would like to know how the training loss evolves for $f\left(\vec{x}\right)$, then an additional call is needed. Hence, there are two versions of code. To obtain the data for loss-vs-iteration, use `Alt_ideal_ghz_q6_b729_d10_sh100_eps3_plotloss.py`. To save computation resource, use `Alt_ideal_ghz_q6_b729_d10_sh100_eps3.py`.
