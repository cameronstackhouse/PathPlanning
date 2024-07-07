"""
TODO
"""

from Sampling_based_Planning.rrt_3D.mb_guided_srrt_edge3D import MbGuidedSrrtEdge


class DynamicGuidedSrrtEdge(MbGuidedSrrtEdge):
    def __init__(self, t=0.1, m=10000):
        super().__init__(t, m)

    def Main(self):
        self.x0 = tuple(self.env.goal)
        self.xt = tuple(self.env.start)

        # Find initial path
        path = self.run()
        self.done = True
        t = 0

        while True:
            # TODO
            pass
