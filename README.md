# gym-el-farol
The El Farol Bar problem is a framework to understand bounded rationality in economics.  In the problem 100 people want to visit a bar every Thursday, but won't be happy if it's too crowded.  Therefore, they're happiest if they visit the bar, and less than 60 other people do so.  Are intermediately happy if they stay home, and are unhappy if they visit the bar and more than 60 others do so.

If agents are perfectly rational there are a huge number of nash equilibria (100 choose 60 pure equilibria alone), which is implausible.  Instead, we need to explore what equilbria occur when agents use different cognition models.  For instance, if agents use q-learning they achieve a pure nash equilibria.  If they use learning on a set of naive predition models a mixed nash equilbria is achieved.  See [Whitehead](http://www.econ.ed.ac.uk/papers/id186_esedps.pdf) for more information.
