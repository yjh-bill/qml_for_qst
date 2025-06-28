## QST for 6-qubit spin-chain ground state

#### Note

By default, for each iteration, SPSA algorithm evaluates $f(\vec{x}+\vec{\Delta})$ and $f(\vec{x}-\vec{\Delta})$, making two function calls. However, if one would like to know how the training loss evolves for $f(\vec{x})$, then an additional call is needed. Hence, there are two versions of code.  To obtain the data for loss-vs-iteration, use `Alt_kl_eps3_ns_plot.py`. To save computation resource, use `Alt_kl_eps3_ns.py`.
