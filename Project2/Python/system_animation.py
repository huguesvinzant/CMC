""" Lab 5 System Animation"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class SystemAnimation(object):
    """ SystemAnimation

    """

    def __init__(
            self,
            res_sys,
            pendulum_sys,
            muscle_sys,
            neural_sys=None,
            fps=50
    ):
        super(SystemAnimation, self).__init__()

        self.pendulum_sys = pendulum_sys
        self.muscle_sys = muscle_sys
        self.neural_sys = neural_sys
        self.time = res_sys[:, 0]
        self.state = res_sys[:, 1:]

        # Define positions for neurons
        self.neurons_pos = np.array([[-.5, .5],
                                     [.5, .5],
                                     [-0.25, 0.25],
                                     [0.25, 0.25]])
        self.fps = fps
        self.fig, self.ax = plt.subplots(num="Simulation")

        self.anims = self.animation_objects()
        t_max = self.time[-1]
        dt = 1 / float(fps)

        self.anim_link = animation.FuncAnimation(
            self.fig, self._animate, np.arange(0, t_max, dt),
            interval=1e3 / float(fps), blit=True
        )
        plt.title("Simulation animation")
        plt.axis('scaled')
        plt.axis('off')
        limit = 1.15 * self.pendulum_sys.parameters.L
        if limit < 0.5:
            limit = .5
        plt.axis([-limit, limit,
                  -limit, 0.75])
        plt.grid(False)
        return

    def animation_objects(self):
        """ Create and return animation objects """

        blue = (0.0, 0.3, 1.0, 1.0)
        # Pendulum
        pendulum = self.pendulum_sys.pose()
        self.line, = self.ax.plot(
            pendulum[:, 0],
            pendulum[:, 1],
            color=blue,
            linewidth=5,
            animated=True
        )
        # Mass
        self.m, = self.ax.plot(
            self.pendulum_sys.origin[0], self.pendulum_sys.parameters.L,
            color=blue, marker='o', markersize=12.5, animated=True)
        # Base
        self.ax.plot([-0.5, 0.5], self.pendulum_sys.origin,
                     c='g', linewidth=7.5)
        # Muscles
        musc = self.muscle_sys.position_from_angle(self.state[0, 0])

        muscles = [self.ax.plot(m[:, 0], m[:, 1], color='r', linewidth=3.5,
                                animated=True)[0]
                   for m in musc]

        # Time
        time = self.ax.text(-0.5, 0.05, "Time: 0.0",
                            fontsize=14, animated=True)

        # Neurons
        if self.neural_sys is not None:
            neurons = [self.ax.scatter(
                self.neurons_pos[:, 0], self.neurons_pos[:, 1],
                s=np.ones(4) * 250, c='r', animated=True)]
            return [self.line, self.m] + muscles + [time] + neurons
        return [self.line, self.m] + muscles + [time]

    @staticmethod
    def animate():
        """Animate System"""
        plt.show()
        return

    def _animate(self, time):
        """ Animation """
        index = np.argmin((self.time - time)**2)
        self.pendulum_sys.theta = self.state[index, 0]
        pendulum = self.pendulum_sys.pose()

        # Pendulum
        self.anims[0].set_xdata(pendulum[:, 0])
        self.anims[0].set_ydata(pendulum[:, 1])

        # Mass
        self.anims[1].set_xdata([pendulum[1, 0]])
        self.anims[1].set_ydata([pendulum[1, 1]])

        # Muscles
        muscles = self.muscle_sys.position_from_angle(self.state[index, 0])
        activations = [self.state[index, 2], self.state[index, 4]]
        for i, musc in enumerate(self.anims[2:4]):
            musc.set_color((activations[i], 0.0, 0.0, 1.0))
            musc.set_xdata(muscles[i][:, 0])
            musc.set_ydata(muscles[i][:, 1])

        # Text
        self.anims[4].set_text("Time: {:.1f}".format(self.time[index]))

        # Neurons
        if self.neural_sys is not None:
            n_rate = self.neural_sys.n_act(self.state[index, 6:])
            self.anims[5].set_color(
                np.array([[0.0, n_rate[0], 0.0, 1.0],
                          [0.0, n_rate[1], 0.0, 1.0],
                          [0.0, n_rate[2], 0.0, 1.0],
                          [0.0, n_rate[3], 0.0, 1.0]]))
            # self.anims[5].set_sizes(np.ones(4) * 250)
            self.anims[5].set_offsets(self.neurons_pos)
        return self.anims

