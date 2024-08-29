from collections import deque
from typing import Deque, List

import numpy as np

from random import shuffle


class Packet():
    def __init__(self, length_bits):
        self.hops_addr = []
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
    
    def gen_time_us(self):
        return self.hops_arrive_time_us[0]
    
    def tot_time_us(self):
        assert self.hops_depart_time_us[-1] >= 0
        return self.hops_depart_time_us[-1] - self.hops_arrive_time_us[0]
    
    def liv_time_us(self, cur_time_us):
        return cur_time_us - self.hops_arrive_time_us[0]

class P2PLinkInterfaceRx():
    def push(self, packets):
        pass

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
   
class PacketSrc():
    def __init__(self, packet_length_bits, packet_rate_per_step):
        self.packet_length_bits = packet_length_bits
        self.packet_rate_per_step = packet_rate_per_step

    def set_rate(self,packet_rate_per_step):
        self.packet_rate_per_step = packet_rate_per_step
        
    def pop(self):
        num_packets = np.random.poisson(self.packet_rate_per_step)

        return [Packet(self.packet_length_bits) for i in range(num_packets)]

class PacketDst():
    def __init__(self, bits_per_step = 0):
        self.packet_counter = 0
        self.pdelay_counter = 0.
        self.pltbit_counter = 0
        
        self.bits_per_step = bits_per_step

    def push(self, packets):
        self.packet_counter += len(packets)
        for p in packets:
            self.pdelay_counter += p.tot_time_us() 
            self.pltbit_counter += p.length_bits

    def pop_from_queue(self, q:Queue):
        tx_bits = 0 
        while not q.is_empty() and tx_bits < self.bits_per_step:
            packets = q.pop(1)
            self.push(packets)
            for p in packets:
                tx_bits += p.length_bits                
        return tx_bits


class P2PLinkDirtTo(P2PLinkInterfaceRx,EnvObject):
    def __init__(self, to_node=None, bits_per_us=1e4, step_time_us=100):
        self.bits_per_us = bits_per_us
        self.step_time_us = step_time_us
        
        self.step_tx_bits_max = self.step_time_us * self.bits_per_us
            
        self.forwarding_packets = []
        
        self.to_node = to_node
        assert isinstance(self.to_node,P2PLinkInterfaceRx)
        
        self.n_discard = 0
        
    def step(self, step_time_us = 100):
        shuffle(self.forwarding_packets)
        
        packets = []
        tx_bits = 0
        while tx_bits <= self.step_tx_bits_max:
            p = self.forwarding_packets.pop(0)
            tx_bits += p.length_bits
            packets.append(p)
        
        self.to_node.push(packets)
        
        self.n_discard += len(self.forwarding_packets)
        self.forwarding_packets = []
        
    def push(self, packets):
        self.forwarding_packets.append(packets)


class NodeWithLocalPacketSrcDst(P2PLinkInterfaceRx):
    def __init__(self, id = 0, queue_size=1000, packet_length_bits = 5000, packet_arrive_rate_per_step = 1000, packet_leave_rate_per_step = 1000):
        self.step_time_us = 1000
        
        self.packet_src = PacketSrc(packet_length_bits=packet_length_bits,packet_rate_per_step=packet_arrive_rate_per_step)
        self.packet_dst = PacketDst(bits_per_step = packet_leave_rate_per_step*packet_length_bits)
        self.packet_que = Queue(max_size=queue_size)
        self.p2plink_to_neighbor = []
        self.p2plink_to_local = P2PLinkDirtTo(to_node=self.packet_que)

    def step(self):
        pass
    
    def push(self, packets):
        self.p2plink_to_local.push(packets)
        
    def multiplexer(self, packets):
        pass

