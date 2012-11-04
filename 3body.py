#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Berechne 3-Körper-Problem-Trajektorien numerisch und animiere die Lösungen.

(c) 2012 Alexander Eberspächer
"""

import numpy as np
from scipy.integrate import ode
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpltools.animation import Animation
from math import sqrt

# definiere ein paar "helper functions":
def L(xVals, yVals, xDotVals, yDotVals):
    """Drehimplus L als Funktion von Ort und Geschwindigkeit.

    Parameter
    ---------
    xVals, yVals : arrays
        $x,y$-Koordinaten $x_i, y_i$
    xDotVals, yDotVals : arrays
        Zeit-Ableitungen $\dot{x}_i$, $\dot{y}_i$

    Bemerkung
    ---------
    Mit den vielen Argumenten lässt sich der Drehimpuls übersichtlich
    mit Hilfer zweier Skalarprokukte schreiben.
    """

    return 1.0/3.0*(np.dot(yVals, xDotVals) + np.dot(xVals, yDotVals))  # m = 1/3


def E(xVals, yVals, xDotVals, yDotVals):
    """Drehimplus L als Funktion von Ort und Geschwindigkeit.

    Parameter
    ---------
    xVals, yVals : arrays
        $x,y$-Koordinaten $x_i, y_i$
    xDotVals, yDotVals : arrays
        Zeit-Ableitungen $\dot{x}_i$, $\dot{y}_i$
    """

    # m_i = 1/3 beachten:

    T = 1.0/6.0*(np.sum(xDotVals**2) + np.sum(yDotVals**2))  # kinetic energy
    V = -1.0/9.0*( 1/np.sqrt((xVals[0] - xVals[1])**2 + (yVals[0] - yVals[1])**2) +
                   1/np.sqrt((xVals[1] - xVals[2])**2 + (yVals[1] - yVals[2])**2) +
                   1/np.sqrt((xVals[0] - xVals[2])**2 + (yVals[0] - yVals[2])**2)
                  )
    return T+V


def grav(x1, y1, x2, y2, m1=1.0/3.0, m2=1.0/3.0):
    """Berechne Gravitationskraft von m2 auf m1.
    """

    # berechne Richtung r2-r1:
    length = sqrt((x2-x1)**2 + (y2-y1)**2)
    direction = np.array([x2-x1, y2-y1])
    direction /= length  # auf Einheitslänge normieren

    return m1*m2/length**2*direction


def xiDot(t, xi):
    r"""Berechne $\dot\xi$ für $\xi = (x_i, y_i, \dot{x_i}, \dot{y_i}).

    Die numerischen Integratoren benötigen Gleichungen erster Ordnung, das
    Newtonsche Kraftgesetz ist aber zweiter Ordnung in der Zeit. Wir reduzieren
    deshalb die Ordnung der Gleichung durch Einführung neuer Variablen.
    """

    # xi = (x1, x2, x3, y1, y2, y3, dotx1, dotx2, dotx2, dotx3, doty1, doty2, doty3)
    # dotXi = (dotx1, dotx2, dotx3, doty1, doty2, doty3,
    #          dotdotx1, dotdotx2, dotdotx3, dotdoty1, dotdoty2, dotdoty3)

    retVal = np.zeros(12)  # Array für xiPunkt
    xVals = xi[0:3]
    yVals = xi[3:6]
    F1 = grav(xVals[0], yVals[0], xVals[1], yVals[1]) + grav(xVals[0], yVals[0], xVals[2], yVals[2])  # Kräfte auf Masse 1 = F21 + F31
    F2 = grav(xVals[1], yVals[1], xVals[0], yVals[0]) + grav(xVals[1], yVals[1], xVals[2], yVals[2])  # Kräfte auf Masse 2 = F12 + F32
    F3 = grav(xVals[2], yVals[2], xVals[0], yVals[0]) + grav(xVals[2], yVals[2], xVals[1], yVals[1])  # Kräfte auf Masse 2 = F13 + F23

    # \ddot{x}, \ddot{y}

    retVal[0:6] = xi[6:12]  # \dot{x}, \dot{y} unverändert wieder ausgeben
    retVal[6] = F1[0]  # \ddot{x_1}
    retVal[7] = F2[0]  # \ddot{x_2}
    retVal[8] = F3[0]  # \ddot{x_3}
    retVal[9] = F1[1]
    retVal[10] = F2[1]
    retVal[11] = F3[1]
    retVal[6:12] *= 3.0  # / (1/3) für \ddot{x} = F/m (wegen m=1/3)!

    return retVal


class Animate(Animation):
    """3-Körper-Animation.
    """

    def __init__(self, integrator, xi0, t0, tstep, tmax, plotEvery=1):
        """Animations-Objekt erzeugen

        Parameters
        ----------
        integrator : object
            Integrator-Objekt (Instanz von scipy.integrate.ode).
        t0, tstep, tmax : double
            Anfangszeit, Zeit-Schritt und Maximal-Zeit.
        xi0 : array
            Vektor mit Anfangsbedingungen gemäß den Konventionen von xiDot.
        """

        self.integrator = integrator

        # Integrator initialisieren:
        self.integrator.set_initial_value(xi0, t0)
        self.tstep = tstep
        self.tmax = tmax
        self.plotEvery = plotEvery

        # Plot vorbereiten:
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.tight_layout()

    def update(self):
        """Plot updaten und DGL integrieren.

        Für bessere Sichtbarkeit werden "Schatten" gezeichnet.
        """

        shadowPoints = 70  # so viele "Schatten" zeichnen

        xVals = self.integrator.y[0:3]
        yVals = self.integrator.y[3:6]

        # "Schatten" = Anfangsorte
        x1Shadows = xVals[0]*np.ones(shadowPoints)
        y1Shadows = yVals[0]*np.ones(shadowPoints)
        x2Shadows = xVals[1]*np.ones(shadowPoints)
        y2Shadows = yVals[1]*np.ones(shadowPoints)
        x3Shadows = xVals[2]*np.ones(shadowPoints)
        y3Shadows = yVals[2]*np.ones(shadowPoints)

        self.line1, = self.ax.plot(x1Shadows, y1Shadows, c="b", lw=1,
                                      marker="o", label="Planet 1")
        self.line2, = self.ax.plot(x2Shadows, y2Shadows, c="r", lw=1,
                                      marker="o", label="Planet 2")
        self.line3, = self.ax.plot(x3Shadows, y3Shadows, c="g", lw=1,
                                      marker="o", label="Planet 3")

        # Bildauschnitt größer machen:
        oldXlim = self.ax.get_xlim()
        oldYlim = self.ax.get_ylim()
        self.ax.set_xlim((3*oldXlim[0], 3*oldXlim[1]))
        self.ax.set_ylim((3*oldYlim[0], 3*oldYlim[1]))
        self.ax.legend(numpoints=3)

        steps = 0
        while(self.integrator.t < self.tmax and self.integrator.successful()):
            # Alle plotEvery Schritte plotten:
            if(steps % self.plotEvery == 0):
                print("t = %s, xi = %s"%(self.integrator.t, self.integrator.y))
                xVals = self.integrator.y[0:3]
                yVals = self.integrator.y[3:6]
                xDotVals = self.integrator.y[6:9]
                yDotVals = self.integrator.y[9:12]
                print("L = %s; E = %s"%(L(xVals, yVals, xDotVals, yDotVals), E(xVals, yVals, xDotVals, yDotVals)))

                x1Shadows = np.roll(x1Shadows, -1)
                x1Shadows[-1] = xVals[0]
                x2Shadows = np.roll(x2Shadows, -1)
                x2Shadows[-1] = xVals[1]
                x3Shadows = np.roll(x3Shadows, -1)
                x3Shadows[-1] = xVals[2]
                y1Shadows = np.roll(y1Shadows, -1)
                y1Shadows[-1] = yVals[0]
                y2Shadows = np.roll(y2Shadows, -1)
                y2Shadows[-1] = yVals[1]
                y3Shadows = np.roll(y3Shadows, -1)
                y3Shadows[-1] = yVals[2]

                self.line1.set_xdata(x1Shadows)
                self.line1.set_ydata(y1Shadows)
                self.line2.set_xdata(x2Shadows)
                self.line2.set_ydata(y2Shadows)
                self.line3.set_xdata(x3Shadows)
                self.line3.set_ydata(y3Shadows)

                yield self.line1, self.line2, self.line3,

            # Schritt machen
            self.integrator.integrate(self.integrator.t + self.tstep)
            steps += 1


if(__name__ == '__main__'):
    # Anfangsbedingunen aus Simo-Paper:
    x3Punkt = 0.7494421910777922289898659
    y3Punkt = 1.1501789857502275024030202
    xDot = np.array([-x3Punkt/2, -x3Punkt/2, x3Punkt])
    yDot = np.array([-y3Punkt/2, -y3Punkt/2, y3Punkt])

    # zu bestimmen: x1, x3:
    x3 = 0.0
    # nutze L(...) = 0 und E + 0.5 = 0 (siehe Paper)
    x, infodict, ier, mesg = fsolve(lambda r: np.array( [L(xVals=(r[0], r[1], x3), yVals=np.zeros(3),
                                                           xDotVals=xDot, yDotVals=yDot),
                                                         E(xVals=(r[0], r[1], x3), yVals=np.zeros(3),
                                                           xDotVals=xDot, yDotVals=yDot) + 0.5] ),  # E = -0.5!
                                         x0=(-0.1, +0.1), full_output=True)
    if(ier is not 1):
        raise ValueError("Keine Startwerte gefunden!")
    else:
        x1 = x[0]; x2 = x[1]
        print("Startpositionen x_1 = %s; x_2 = %s"%(x1, x2))

    integrator = ode(xiDot)  # Integrator erzeugen
    integrator.set_integrator("dop853")  # Dormand-Price-Integration

    # Anfangsvektor xi0 erzeugen:
    xi0 = np.array([x1, x2, x3, 0.0, 0.0, 0.0, xDot[0], xDot[1], xDot[2], yDot[0], yDot[1], yDot[2]])

    Animator = Animate(integrator, xi0=xi0, t0=0.0, tstep=0.001, tmax=100.0, plotEvery=7)
    Animator.animate(blit=False, interval=35.0)
    plt.show()
