from tmfc_simulation.wc_exploration import simple_one_node, ito_version
import matplotlib.pyplot as plt

def test_simple_one_node():
    exc_ext = 1.5
    wc = simple_one_node(exc_ext=exc_ext)

    plt.plot(wc.exc[0, -int(1000 / wc.params['dt']):])
    plt.show()
    assert False

def test_ito_version():
    K_gl = 4
    exc_ext = -5
    wc = ito_version(exc_ext=exc_ext)
    assert False

