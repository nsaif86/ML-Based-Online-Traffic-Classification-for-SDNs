#!/usr/bin/python
#Description: This code was written in python to generate traffic data
#which is used to train a classification model.

#import the packages
import subprocess, sys #to handle the output of the monitoring model
import signal #f or timer
import os #to deal with operating system
import numpy as np #for model features
import time #to synchronize time with the monitoring model
from prettytable import PrettyTable #to display output from ML model
import pickle #to use ML model real-time
import tensorflow as tf

flows = {} #dictionary to collect flows

TIMEOUT = 60 * 50 #6000 min,how long to collect training data

cmd = "sudo ryu run /home/mohammed/ryu/ryu/app/classifier/monitor.py"#run monitoring model as a model within Ryu controller

#create a class to collect and update the training data
class Flow:
    def __init__(self, time_start, datapath, ethsrc, ethdst, inport, outport, packets, bytes):
    
        #setting the essential properties
        self.time_start = time_start
        self.datapath = datapath
        self.ethsrc = ethsrc
        self.ethdst = ethdst
        self.inport = inport
        self.outport = outport
        
        #properties for forward flow direction (src -> dst)
        self.forward_packets = packets
        self.forward_bytes = bytes
        self.forward_delta_packets = 0
        self.forward_delta_bytes = 0
        self.forward_inst_pps = 0.00
        self.forward_inst_bps = 0.00
        self.forward_avg_pps = 0.00
        self.forward_avg_bps = 0.00
        self.forward_status = 'ACTIVE'
        self.forward_last_time = time_start
        
        #properties for reverse flow direction (destination -> source)
        self.reverse_packets = 0
        self.reverse_bytes = 0
        self.reverse_delta_packets = 0
        self.reverse_delta_bytes = 0
        self.reverse_inst_pps = 0.00
        self.reverse_inst_bps = 0.00
        self.reverse_avg_pps = 0.00
        self.reverse_avg_bps = 0.00
        self.reverse_status = 'INACTIVE'
        self.reverse_last_time = time_start
        
    #updates the properties in the forward flow direction
    def updateforward(self, packets, bytes, curr_time):
        # Packets
        self.forward_delta_packets = packets - self.forward_packets
        self.forward_packets = packets
        if curr_time != self.time_start: self.forward_avg_pps = packets/float(curr_time-self.time_start)
        if curr_time != self.forward_last_time: self.forward_inst_pps = self.forward_delta_packets/float(curr_time-self.forward_last_time)
        # Bytes
        self.forward_delta_bytes = bytes - self.forward_bytes
        self.forward_bytes = bytes
        if curr_time != self.time_start: self.forward_avg_bps = bytes/float(curr_time-self.time_start)
        if curr_time != self.forward_last_time: self.forward_inst_bps = self.forward_delta_bytes/float(curr_time-self.forward_last_time)
        self.forward_last_time = curr_time
        if (self.forward_delta_bytes==0 or self.forward_delta_packets==0): #if the flow did not receive any packets of bytes
            self.forward_status = 'INACTIVE'
        else:
            self.forward_status = 'ACTIVE'
        
    #updates the attributes in the reverse flow direction
    def updatereverse(self, packets, bytes, curr_time):
        # Packets
        self.reverse_delta_packets = packets - self.reverse_packets
        self.reverse_packets = packets
        if curr_time != self.time_start: self.reverse_avg_pps = packets/float(curr_time-self.time_start)
        if curr_time != self.reverse_last_time: self.reverse_inst_pps = self.reverse_delta_packets/float(curr_time-self.reverse_last_time)
        # Bytes
        self.reverse_delta_bytes = bytes - self.reverse_bytes
        self.reverse_bytes = bytes
        if curr_time != self.time_start: self.reverse_avg_bps = bytes/float(curr_time-self.time_start)
        if curr_time != self.reverse_last_time: self.reverse_inst_bps = self.reverse_delta_bytes/float(curr_time-self.reverse_last_time)
        self.reverse_last_time = curr_time
        if (self.reverse_delta_bytes==0 or self.reverse_delta_packets==0): #if the flow did not receive any packets of bytes
            self.reverse_status = 'INACTIVE'
        else:
            self.reverse_status = 'ACTIVE'

#function to print flow properties when collecting training data
def export_flows(traffic_type,f):
    for key,flow in flows.items():
        outstring = '\t'.join([
        str(flow.forward_packets),
        str(flow.forward_bytes),
        str(flow.forward_delta_packets),
        str(flow.forward_delta_bytes), 
        str(flow.forward_inst_pps), 
        str(flow.forward_avg_pps),
        str(flow.forward_inst_bps), 
        str(flow.forward_avg_bps), 
        str(flow.reverse_packets),
        str(flow.reverse_bytes),
        str(flow.reverse_delta_packets),
        str(flow.reverse_delta_bytes),
        str(flow.reverse_inst_pps),
        str(flow.reverse_avg_pps),
        str(flow.reverse_inst_bps),
        str(flow.reverse_avg_bps),
        str(traffic_type)])
        f.write(outstring+'\n')
        
        
#function to print classifyier out put
def classifier_output(model):
    x = PrettyTable()
    x.field_names = ["Flow ID", "Src MAC", "Dest MAC", "Traffic Type","Forward Status","Reverse Status"]
    #print ("hi")
    for key,flow in flows.items():
        features = np.asarray([flow.forward_delta_packets,flow.forward_delta_bytes,flow.forward_inst_pps,flow.forward_avg_pps,flow.forward_inst_bps,flow.forward_avg_bps,
        flow.reverse_delta_packets,flow.reverse_delta_bytes,flow.reverse_inst_pps,flow.reverse_avg_pps,flow.reverse_inst_bps,
        flow.reverse_avg_bps,flow.forward_packets,flow.forward_bytes,flow.reverse_packets,flow.reverse_bytes]).reshape(1,-1) #input features
        #print ("hiiiiiiiiiiiiiiiiiiiiiiiii1",features)
        #print ("hiiiiiiiiiiiiiiiiiiiiiiiii1",features.shape)
 
        label = np.argmax(model.predict(features.tolist()),axis=-1) #if model is LR then the label is the type of traffic
       
        #print ("label",label)
        x.add_row([key, flow.ethsrc, flow.ethdst, label[0],flow.forward_status,flow.reverse_status]) 
    print(x)#print output
    #print ("hiiiiiiiiiiiiiiiiiiiiiiiii3")       
        
        
        

# build function to process the output of the monitoring model        
def monitor_output_record(p,traffic_type=None,f=None, model=None):
    time = 0
    while True:
        #record going through loop untill time out
        output = p.stdout.readline()# recivre the output data in variable
        #print ("output", output)
        if output == '' and p.poll() != None:    # check if there is data
            break 
        if output != '' and output.startswith(b'data'): #when monitoring model returns output
            #print ("output.startswith(b'data')", output.startswith(b'data'))
            fields = output.split(b'\t')[1:] #split the flow details
            #print ("fields", fields)
            fields = [f.decode(encoding='utf-8', errors='strict') for f in fields] #decode flow details             
            uni_id = hash(''.join([fields[1],fields[3],fields[4]])) #create unique ID for flow based on switch ID, source host,and destination host
            #print ("uni_id", uni_id)
            if uni_id in flows.keys():
                flows[uni_id].updateforward(int(fields[6]),int(fields[7]),int(fields[0])) #update forward properties with time, packet, and byte count
            else:
                rev_uni_id = hash(''.join([fields[1],fields[4],fields[3]])) #switch source and destination to generate same hash for src/dst and dst/src
                #print ("rev_uni_id", rev_uni_id)
                if rev_uni_id in flows.keys():
                    flows[rev_uni_id].updatereverse(int(fields[6]),int(fields[7]),int(fields[0])) #update reverse properties with time, packet, and byte count
                else:
                    flows[uni_id] = Flow(int(fields[0]), fields[1], fields[2], fields[3], fields[4], fields[5], int(fields[6]), int(fields[7])) #create new flow object
            if not model is None:
                if time%20==0: #show the output of model 
                    classifier_output(model)
                    #print ("hiiiiiiiiiiiiiiiiiiiiiiiii") 
            else:
                  export_flows(traffic_type,f) #for export training data to csv file
    time += 1
 
#print help output in case of incorrect options 
def printHelp():
    print("Usage: sudo python data.py [subcommand] [options]")
    print("\tTo collect training data for a certain type of traffic, run: sudo python data.py train [voice_G.xx|video|dns|telnet|ping|game_idle|game_active]")
    return

#for timer to collect flow training data
def alarm_handler(signum, frame):
    print("Signal Number:", signum, " Frame: ", frame)  
    print('Alarm at:', time.ctime())
    print("Finished collecting data.")
    raise Exception()
    
if __name__ == '__main__':
    SUBCOMMANDS = ('train', 'LR','NN','SVM')

    if len(sys.argv) < 2:
        print("ERROR: Incorrect # of args")
        print()
        printHelp()
        sys.exit();
    else:
        if len(sys.argv) == 2:
            if sys.argv[1] not in SUBCOMMANDS:
                print("ERROR: Unknown subcommand argument.")
                print("Currently subaccepted commands are: %s" % str(SUBCOMMANDS).strip('()'))
                print()
                printHelp()
                sys.exit();

    if len(sys.argv) == 1:
        # Called with no arguments
        printHelp()
    elif len(sys.argv) >= 2:
        if sys.argv[1] == "train":
            if len(sys.argv) == 3:
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) #start monitor model
                time.sleep(15) # waiting time to upload monitoring model

                traffic_type = sys.argv[2]
                f = open(traffic_type+'_training_data.csv', 'w') #open training data output file
                signal.signal(signal.SIGALRM, alarm_handler) #signal process to start and end colecting training data
                signal.alarm(TIMEOUT) #set for 100 minutes
                print('Current time:', time.ctime())

                try:
                    headers = 'Forward Packets\tForward Bytes\tDelta Forward Packets\tDelta Forward Bytes\tForward Instantaneous Packets per Second\tForward Average Packets per second\tForward Instantaneous Bytes per Second\tForward Average Bytes per second\tReverse Packets\tReverse Bytes\tDelta Reverse Packets\tDelta Reverse Bytes\tDeltaReverse Instantaneous Packets per Second\tReverse Average Packets per second\tReverse Instantaneous Bytes per Second\tReverse Average Bytes per second\tTraffic Type\n'
                    f.write(headers)
                    monitor_output_record(p,traffic_type=traffic_type,f=f)
                except Exception:
                    print('Exiting')
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM) #terminate and exit
                    f.close()
            else:
                print("ERROR: specify traffic type.\n")
                printHelp()
        else:
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) #start ryu process
            time.sleep(15) # waiting time to upload monitoring model
            if sys.argv[1] == 'LR':
                infile = open('LR','rb') 
                print ("model is runing_RL")
                model = pickle.load(infile) #load ML model 
            elif sys.argv[1] == 'NN':

                print ("model is runing_NN")
                model = tf.keras.models.load_model('NN_2')
            elif sys.argv[1] == 'SVM':
                infile = open('SVM','rb')
                
                print ("model is runing_SVM")
                model = pickle.load(infile)

            #print ("hiiiiiiiiiiiiiiiiiiiiiiiii0") 
            monitor_output_record(p,model=model)
    sys.exit();
