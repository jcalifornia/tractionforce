import numpy as np

nu = 0.5
E = 1

def fxx(x,y,l2):
    """

    :param x:
    :param y:
    :param l2:  || x - y || L2 distance so we don't need to recompute
    :return:
    """
    return (nu+1)/np.pi/E*( x*(1-nu)*np.log(l2+y)+y*np.log(l2+x)-y )

def fxxx(x,y,l2):
    return (nu+1)/(2*np.pi*E)*((nu+1)*y*l2 - (nu-1)*x**2*np.log(l2 + y))

def fxxy(x,y,l2):
    return (nu+1.0)/(2*np.pi*E)*( y**2*np.log(l2+x) - l2*( (2*nu-1)*x + 0.5*l2))

def fxy(x,y,l2):
    return -nu*(nu+1)/(np.pi*E)*l2

def fxyx(x,y,l2):
    return nu*(nu+1)/(2*np.pi*E)*(y**2*np.log(l2+x)-0.5*l2*(l2+2*x) )
