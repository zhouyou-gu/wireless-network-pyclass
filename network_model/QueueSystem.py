from collections import deque
from typing import Deque, List

import numpy as np

from random import shuffle

import simpy

class Packet():
    def __init__(self, length_bits, gen_time_us):
        self.hops_addr = []
        self.gen_time_us = gen_time_us
        self.end_time_us = -1
        self.hops_arrive_time_us = []
        self.hops_depart_time_us = []

        self.length_bits = length_bits
        
    def arrive_at(self, addr, cur_time_us):
        self.hops_addr.append(addr)
        self.hops_arrive_time_us.append(cur_time_us)
        self.hops_depart_time_us.append(-1)

    def depart_at(self, addr, cur_time_us):
        assert self.hops_addr[-1] == addr
        self.hops_depart_time_us[-1] = cur_time_us
        
    def get_last_hop_arr_time(self):
        return self.hops_arrive_time_us[-1]
    
    def get_last_hop_que_time(self, cur_time_us):
        return cur_time_us - self.hops_arrive_time_us[-1]
    
    def get_gen_time_us(self):
        return self.gen_time_us
    
    def get_tot_time_us(self):
        assert self.end_time_us >= 0
        return self.end_time_us - self.gen_time_us
    
    def get_liv_time_us(self, cur_time_us):
        return cur_time_us - self.gen_time_us
    
    def end(self,cur_time_us):
        self.end_time_us = cur_time_us
        print(self.gen_time_us,self.end_time_us)

        return self.get_tot_time_us()

class P2PLinkInterfaceRx():
    def push(self, packets):
        pass

# class P2PLinkInterfaceTx():
#     def push(self, packets):
#         pass

class Queue(P2PLinkInterfaceRx):
    def __init__(self, id=0, max_size=10000):
        self.id = id
        self.n_packet = 0
        self.queue:Deque[Packet] = deque(maxlen=max_size)

    def get_hol_vs_hop(self, cur_time_us):
        if self.queue:
            return cur_time_us - self.queue[0].get_last_hop_que_time(self, cur_time_us)
        else:
            return 0

    def get_hol_vs_src(self, cur_time_us):
        if self.queue:
            return cur_time_us - self.queue[0].liv_time_us(self, cur_time_us)
        else:
            return 0
    
    def pop(self, n=1):
        ret = []
        while self.queue and n > 0:
            ret.append(self.queue.popleft())
            n -= 1
        return ret
    
    def push(self, packets):
        n_discard = 0
        for p in packets:
            if len(self.queue) == self.queue.maxlen:
                n_discard += 1
            else:
                self.queue.append(p)
        return n_discard
    
    def get_bits_total(self):
        ret = 0
        for x in range(len(self.queue)):
            ret += self.queue[x].length_bits

        return ret 

    def get_pkts_total(self):
        return len(self.queue)
    
    def is_empty(self):
        return len(self.queue) == 0 
   
class EnvObject():
    env = None
    @classmethod
    def set_env(cls,env):
        cls.env = env

class EnvObjectRunnable(EnvObject):
    def __init__(self) -> None:
        EnvObject.env.process(self.run())
    
    def run(self):
        pass

class PacketSrc(EnvObjectRunnable):
    def __init__(self, to_node=None, packet_length_bits=5e3, bit_rate_per_us = 5e6, packet_generation_interval_us = 1e3):
        super().__init__()
        self.packet_length_bits = packet_length_bits
        self.bit_rate_per_us = bit_rate_per_us
        self.packet_generation_interval_us = packet_generation_interval_us

        self.to_node = to_node

        self.n_packet_per_interval = self.bit_rate_per_us*self.packet_generation_interval_us/self.packet_length_bits

    def set_bit_rate(self,bit_rate_per_us):
        self.bit_rate_per_us = bit_rate_per_us

    def run(self):
        while True:
            packets = self._gen_packets()
            self.to_node.push(packets)
            yield self.env.timeout(self.packet_generation_interval_us)            

    def _gen_packets(self):
        num_packets = np.random.poisson(self.n_packet_per_interval)
        packets = [Packet(self.packet_length_bits,self.env.now) for i in range(num_packets)]
        return packets

class PacketDst(EnvObject,P2PLinkInterfaceRx):
    def __init__(self):
        self.packet_counter = 0
        self.pdelay_counter = 0.
        self.pltbit_counter = 0
        
    def push(self, packets:List[Packet]):
        self.packet_counter += len(packets)
        for p in packets:
            self.pdelay_counter += p.end(self.env.now) 
            self.pltbit_counter += p.length_bits

class P2PLinkDirtTo(EnvObject,P2PLinkInterfaceRx):
    def __init__(self, to_node=None, bits_per_us=1e4, delay_us=100):
        self.to_node = to_node
        self.bits_per_us = bits_per_us
        self.delay_us = delay_us
        
        self.n_discard = 0
        self.current_load_bits = 0 
        
    def push(self, packets):
        for p in packets:
            tx_time_us = p.length_bits / self.bits_per_us
            if self.current_load_bits + p.length_bits > self.bits_per_us * tx_time_us:
                self.n_discard += 1
                continue
            else:
                self.env.process(self.push_a_packet(p,tx_time_us))

    def push_a_packet(self, p:Packet, tx_time_us):
        self.current_load_bits += p.length_bits
        yield self.env.timeout(self.delay_us + tx_time_us)
        self.to_node.push([p])
        self.current_load_bits -= p.length_bits
        

class NodeWithLocalPacketSrcDst(EnvObjectRunnable,P2PLinkInterfaceRx):
    def __init__(self, id = 0, queue_size=10, packet_rate_per_ms = 5):
        super().__init__()
        self.id = id
        self.packet_src = PacketSrc(to_node=self)
        self.packet_dst = PacketDst()
        self.packet_que = Queue(max_size=queue_size)
        
        self.packet_rate_per_ms = packet_rate_per_ms
        self.p2plink_to_neighbor = []

    def run(self):
        self.env.process(self.queueing())
        yield env.event()

    def queueing(self):
        while True:
            print(self.packet_dst.packet_counter,self.packet_dst.pdelay_counter,self.packet_dst.pdelay_counter/(self.packet_dst.packet_counter+1))
            yield self.env.timeout(1000)
            self.packet_dst.push(self.packet_que.pop(self.packet_rate_per_ms))
    
    def push(self, packets):
        a = int(len(packets)/2)
        self.packet_que.push(packets[0:a])
        for l in self.p2plink_to_neighbor:
            l.push(packets[a:len(packets)])
        # print(self.packet_que.get_pkts_total())

if __name__ == "__main__":
    import simpy

    class Parent(EnvObjectRunnable):
        def __init__(self,c):
            super().__init__()
            self.c = c
            
        def run(self):
            self.c.hello_out()
            yield self.env.timeout(1)
            print("Parent run",env.now)

    class Child(EnvObject):
        def __init__(self) -> None:
            super().__init__()
            self.counter = 0
        def run(self):
            pass
            while True:
                yield self.env.timeout(1)
                print("Child run")
        
        def hello_out(self):
            print("child hello_out +",self.env.now)
            self.env.process(self.hello(1.1))
            self.env.process(self.hello(1.2))

                    
        def hello(self,a):
            self.counter += 1
            print("child hello +",self.env.now,a,self.counter)
            yield self.env.timeout(a)
            print("child hello -",self.env.now,a,self.counter)
            

    # SimPy environment setup
    env = simpy.Environment()
    EnvObject.set_env(env=env)
    n_a = NodeWithLocalPacketSrcDst()
    n_b = NodeWithLocalPacketSrcDst()
    n_a.p2plink_to_neighbor.append(P2PLinkDirtTo(n_b))
    child_instance = Child()
    parent_instance = Parent(child_instance)
    env.run(until=10000)
    
    print(n_b.packet_dst.packet_counter)
    