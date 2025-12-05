"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
import time
import logging

import nni
import copy
from nni.experiment.remoteExperiment import Experiment
import numpy as np
import json


import numpy as np
import time
from OptimizationTestFunctions import Michalewicz, AckleyTest, Sphere, Rastrigin

from nan.functions import get_pos_mask, get_size, cutoff, denosie, check_gain, check_gain_SUD

from nan.beam_status import wait_beam_status, spark, beam_not_at_SUD_end, Beam_not_at_SBP_end, bad_shutter, bad_interlock, Beam_not_at_Linac_end
from epics import caget, caget_many, caput, caput_many
from nan.epics_wrap import caget_many_wait
from nan.utils import fit_gaussian, get_size_emit, get_size_rad
import os

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
start = time.time()

from threading import Event, Thread
from queue import Queue, Empty
import atexit


experiment_img_dir = 'sud_exp'
def save_img(img_queue):
    while True:
        try:
            id, img = img_queue.get(timeout=1)
            np.save(f'{experiment_img_dir}/obj_{id}', img)
        except Empty:
            pass

_img_queue = Queue()
save_img_thread = Thread(target=save_img, args=(_img_queue,), daemon=True)

save_img_thread.start()
atexit.register(save_img_thread.join)

beam_not_at_SBP_end = Beam_not_at_Linac_end()


gain_scale = 1.0

class ObjBase():
    avg_num = 10
    wait_time = 0.3
    run_id = 0

    def __init__(self, avg_num: int, wait_time: float, report_inter, report_fianl, should_stop) -> None:
        
        self.avg_num = avg_num
        self.wait_time = wait_time
        
        self._opt_sigma = None
        self.current_values = [None]*5

        self.report_inter = report_inter
        self.report_fianl = report_fianl
        self.should_stop = should_stop
        
        
    def __call__(self, x):
        # x = kwargs
        pvs = list(x.keys())
        self.current_values = values = list(x.values())

        caput_many(pvs, values)
        time.sleep(0.5)

        obj = self._objective()

        logger.info(f'Run id: {self.run_id}')
        logger.info(f'Objective: {obj}')
        
        # self.run_id += 1
        return obj
        
    def _objective(self):
        raise NotImplementedError()

class SUD_profile(ObjBase):

    power_pv = 'SUD-BI:PRF17:RAW:ArrayData'
    gain_pv = 'SUD-BI:PRF17:CAM:Gain'
    delay_pv = 'UD-CN:TIM-15A:P2Delay'
    
    def _check_gain(self,img_pv, gain_pv, img=None, avg_num=10, wait_time=0.5):
        check_gain_SUD(img_pv, gain_pv, img, avg_num, wait_time)
    
    def img_process(self,img):
        if caget(self.gain_pv) >= 5:
            img = denosie(img)
        return img

    def get_power_density(self):
        global gain_scale
        # wait_beam_status(reset=True, wait_after_normal=60, 
        #                  check_stop=lambda _: self.stop_flag.token)
        while spark() and beam_not_at_SBP_end():
            print('spark')
            time.sleep(5)
        try:
            img = caget(self.power_pv).reshape((1280, 960), order="F")
        except AttributeError:
                
            print('Profiler error!!!!')
            caput('SUD-BI:SERV14:REBOOT', 1)
            time.sleep(10)
            caput('SUD-BI:SERV14:REBOOT', 0)
            
            while caget(self.power_pv) is None or len(caget(self.power_pv)) == 0:
                time.sleep(10)
                caput('SUD-BI:PRF17:CAM:Acquire', 1)
            print('Profiler recovery!!!!')
            img = caget(self.power_pv).reshape((1280, 960), order="F")
                    
        img_org = copy.deepcopy(img)
        img = self.img_process(img)
        
        size, gaus_mask = get_size_rad(img, ret_gaus_mask=True)
        # pos_mask = get_pos_mask(img, x0=700, y0=520, size=size*3)
        
        img = img * gaus_mask #* pos_mask * 25
        
        return np.sum(img)/size**2, img_org

    def _objective(self):

        power = []
        i=0

        while len(power) == 0:
            i+=1
            power = []
            img = []
            for _ in range(self.avg_num):
                t0 = time.time()
                power_i, img_i = self.get_power_density()
                power.append(power_i)
                img.append(img_i)
                
                # print('wait: ', abs(time.time()-t0))
                time.sleep(max(self.wait_time-abs(time.time()-t0), 0.001))
                self.report_inter(self.run_id,  power_i/(caget(self.gain_pv)**3+1) * caget(self.delay_pv)*1e3 - 1.0)
                
                if self.should_stop(self.run_id):
                    print('Break at Tiral: ', self.run_id)
                    early_stoped = True
                    return early_stoped
                    
            # power = sorted(power)[int(self.avg_num/1.5):]       
            power = np.array(power)
            if i>2 :
                break            
        
        self.gain = caget(self.gain_pv)

        if self.gain >= 0 and self.run_id>=64:
            self._check_gain(self.power_pv, self.gain_pv, img=img, avg_num=self.avg_num, wait_time=self.wait_time)
        if self.gain != caget(self.gain_pv):
            return self._objective()

 
        if not os.path.exists(f'{experiment_img_dir}'):
            os.mkdir(f'{experiment_img_dir}')
        _img_queue.put((self.run_id, np.array(img[-5:])))

        power = power.mean() #+ power.max()
        
        self.gain = caget(self.gain_pv)

        final = power/(caget(self.gain_pv)**3+1) * caget(self.delay_pv)*1e3 - 1.0   # + self.gain*0.5 - caget('UD-CN:TIM-15:P2Delay')*1e3

        return final
    
class SBP_profile(SUD_profile):

    power_pv = 'SBP-BI:PRF16:RAW:ArrayData'
    gain_pv = 'SBP-BI:PRF16:CAM:Gain'

    delay_pv = 'UD-CN:TIM-15:P2Delay'
    
    beam_not_at_SBP = Beam_not_at_Linac_end()
        
    
    def _check_gain(self,img_pv, gain_pv, img=None, avg_num=10, wait_time=0.5):
        check_gain(img_pv, gain_pv, img, avg_num, wait_time)
        
    def get_power_density(self):
        global gain_scale
        
        while self.beam_not_at_SBP():
            time.sleep(5)
            print('Beam not at undulator end!!!!')
            
        # wait_beam_status(reset=True, wait_after_normal=60, 
        #                  check_stop=lambda _: self.stop_flag.token)
        
        while spark():
            print('spark')
            time.sleep(5)
            
        try:
            img = caget(self.power_pv).reshape((1280, 960), order="F")
        except AttributeError:
            
            print('Profiler error!!!!')
            while caget(self.power_pv) is None or len(caget(self.power_pv)) == 0:
                time.sleep(1)
            print('Profiler recovery!!!!')
            img = caget(self.power_pv).reshape((1280, 960), order="F")
            
        img_org = copy.deepcopy(img)
          
        img = self.img_process(img)
        
        size, gaus_mask = get_size_rad(img, ret_gaus_mask=True)
        # size, gaus_mask = get_size(img, ret_gaus_mask=True)
        pos_mask = get_pos_mask(img, x0=660, y0=570, size=size*2)
        
        img = img * gaus_mask * pos_mask
        
        return np.sum(img)/size**2*gain_scale, img_org

if __name__ == '__main__':

    exp = Experiment(None).connect(8071, ip='127.0.0.1')
    exp_status = exp.get_status()
    exp_profile = exp.get_experiment_profile()
    logDir = exp_profile['logDir']
    assert exp_status == 'RUNNING', f'Error! See log file in {logDir}'
    nni.manual_set_experiment(exp)

    experiment_img_dir = exp_profile['id']

    init_params = {pv: caget(pv) for pv in exp_profile['params']['searchSpace'].keys()}
    exp_profile['initialValue'] = init_params

    if not os.path.exists(f'{experiment_img_dir}'):
        os.mkdir(f'{experiment_img_dir}')
    with open(f'{experiment_img_dir}/profile.json', 'w') as f:
        json.dump(exp_profile, f, indent=4)
        
    objective = SBP_profile(avg_num=15, wait_time=0.2, 
                            report_inter=nni.manual_report_intermediate_result, 
                            report_fianl=nni.manual_report_final_result, 
                            should_stop=nni.manualTrial.should_stop_trial)

    def select_params(new_params:dict):
        return {key: new_params.get(key, init_params[key]) for key in init_params.keys()}
    gain_scale = 1e5
    for i in range(2000):
        nni_params = nni.manual_get_next_parameter(i)
        params = copy.deepcopy(init_params)
        params.update(select_params(nni_params))
        
        objective.run_id = i
        result = objective(params)
        
        if not isinstance(result, float):
            continue
        

        nni.manual_report_final_result(i, result)
        print(f'ID:{i}, final result is {result}')
        logger.info(f'ID:{i}, final result is {result}')
        # logger.debug('Send final result done.')

    print('Total run ', time.time()-start)
    input('Press return to stop!')
    exp.stop()




