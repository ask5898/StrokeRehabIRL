
import numpy as np

import gym
from gym import spaces

import os
import opensim as osim

from Networks import PolicyNet
import glob

ACTION = [
            'lumbar_extension','lumbar_bending','lumbar_rotation',
            'arm_flex_r','arm_add_r','arm_rot_r',
            'elbow_flex_r','pro_sup_r'
        ]

OBSERVATION = [
                'stn1', # Chest marker x,y,z
                'shoulder', # x,y,z
                'pelvis_tx',	           'pelvis_ty',	           'pelvis_tz',
                'idx2', #Index Finger marker x,y,z
                'xT_Ball','yT_Ball','zT_Ball', # fixed value
                'EM' #elbow marker x,y,z
              ]

OBSERVATION_LEN = 18
REMOVE = glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\HM*Healthy*.mot'.replace('\\','/'))
N = len(glob.glob(r'C:\Users\ROBOTIC5\Desktop\IRL\Datasets\SharingData_IBEC_VALDUCE_July2021\*Healthy*.mot')) - len(REMOVE)

class PatientSim(gym.Env):    
    def __init__(self, model_path, visualize, motion_paths=None):
        super(PatientSim, self).__init__()
        self.model = osim.Model(model_path)
        self.motions = motion_paths
        self.visualize = visualize
        self.model.setUseVisualizer(self.visualize)
        self.model.setGravity(osim.Vec3(0,0,0))
        self.joint_set = self.model.getJointSet()
        self.actuator_set = self.model.getActuators()
        self.coord_set = self.model.getCoordinateSet()
        self.marker_set = self.model.getMarkerSet()
        self.force_set = self.model.getForceSet()
        self.body_set = self.model.getBodySet()
        self.rem_force = self.model.updForceSet()
        self.sample_expert_action =  lambda i : [action[i] for action in self.expert_actions.values() if action]
        self.removeForces()
        self.state = self.model.initSystem()
        self.action_space = spaces.Box(low= np.array([-1] * len(ACTION)), high= np.array([1] * len(ACTION)), dtype=np.double)
        self.observation_space = spaces.Box(low= np.array([0] * (OBSERVATION_LEN)), high= np.array([0] * (OBSERVATION_LEN), dtype=np.double))  # Joint angles, velocities, and body position (x, y, z)
        self.expert_actions = None
        
        # Simulation parameters
        self.step_size = 0.01  # Time step for simulation (seconds)
        self.max_steps = N*1000  # Max steps per episode
        self.manager = None
        self.current_step = 0
        
    def removeForces(self) :
        for i in reversed(range(self.rem_force.getSize())) :
            self.rem_force.remove(i)
        self.model.finalizeConnections()

    def joinTimeSeriesTable(self) :
        first = True
        for motion in self.motions :
            motionTable = osim.TimeSeriesTable(motion)
            labels, angles = self.expertImitation(motionTable)
            if first :
                self.expert_actions = dict(zip(labels,angles))
            else :
                idx = 0
                for key in self.expert_actions.keys() :
                    self.expert_actions[key].extend(angles[idx])
                    idx = idx + 1

            first = False



    def expertImitation(self, motion) :
        labels = motion.getColumnLabels()
        expertLabels = list()
        expertAngles = list()
        for i in range(len(labels)) :
            coord_name = labels[i]
            if coord_name in ACTION:
                expertLabels.append(coord_name)
                values = motion.getDependentColumn(coord_name)
                angle = [values[i] for i in range(0,values.size())]
                expertAngles.append(angle)

        return expertLabels, expertAngles

    def reset(self, expert=False,seed=None):
        super().reset(seed=seed)
        self.state = self.model.initializeState()
        if expert :
            self.joinTimeSeriesTable()       
        self.manager = osim.Manager(self.model)
        self.manager.initialize(self.state)
        self.current_step = 0
        observation = self.getObservation()
        info = {}
        return observation, info

    def step(self, action):
        # Apply angles to joints
        act = 0
        for i in range(self.coord_set.getSize()) :
            if self.coord_set.get(i).getName() in ACTION :
            
                angle_write = action[act]*np.pi/180
                
                self.coord_set.get(i).setLocked(self.state, False)
                self.coord_set.get(i).setValue(self.state, angle_write)
                act = act + 1
               
        self.model.realizePosition(self.state) 
        self.state =self.manager.integrate(self.current_step * self.step_size)
        observation = self.getObservation()
        reward = 0
        # Check termination conditions
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return observation, reward, terminated

    def getObservation(self):
        obs = list()
        for i in range(self.marker_set.getSize()) :
            marker = self.marker_set.get(i)
            if marker.getName() in OBSERVATION :
                coords = marker.getLocationInGround(self.state)
                obs.extend([coords.get(0),coords.get(1),coords.get(2)])

        for i in range(self.coord_set.getSize()) :
            coord = self.coord_set.get(i)
            if coord.getName() in OBSERVATION :
                obs.append(coord.getValue(self.state))
    
        obs = np.array(obs)
        return obs
    
    def render(self):
        #print('Using OpenSim v4.5 for sumulation')
        pass

    def close(self):
        pass



