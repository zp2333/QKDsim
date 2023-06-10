from qns.entity.cchannel.cchannel import ClassicChannel
from qns.entity.qchannel.qchannel import QuantumChannel
from qns.entity import QNode
from qns.simulator.simulator import Simulator
from qns.network.protocol.bb84 import BB84RecvApp, BB84SendApp
import numpy as np

light_speed = 299791458

#长度单位为米
def drop_rate(length):
    # drop 0.2 db/KM
    return 1 - np.power(10,- length / 50000)

def link_sim(send_rate,length,link_num):
    results = []
    for i in range(link_num):
        s = Simulator(0, 10, accuracy=10000000000)
        n1 = QNode(name="n1")
        n2 = QNode(name="n2")

        qlink = QuantumChannel(name="l1", delay=length / light_speed,
                               drop_rate=drop_rate(length))

        clink = ClassicChannel(name="c1", delay=length / light_speed)

        n1.add_cchannel(clink)
        n2.add_cchannel(clink)
        n1.add_qchannel(qlink)
        n2.add_qchannel(qlink)

        sp = BB84SendApp(n2, qlink, clink, send_rate)
        rp = BB84RecvApp(n1, qlink, clink)
        n1.add_apps(sp)
        n2.add_apps(rp)

        n1.install(s)
        n2.install(s)

        s.run()
        results.append(len(rp.succ_key_pool)/10)
    return np.sum(results)
