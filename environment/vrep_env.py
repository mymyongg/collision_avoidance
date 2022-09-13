from .env_modules import vrep
import numpy as np
import time

class Env():
    def __init__(self, port):
        self.clientID = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)
        self.ego_handles = {}
        self.risky_handles = {}
        self.rear_handles = {}
        self.lateral_handles = {}
        self.left0_handles = {}
        self.left1_handles = {}
        self.opposite_handles = {}
        self.right0_handles = {}
        self.collision_handles = {}
        self.lidar_handles = {}

        # Vehicle lists
        # self.vehicle_list = ['EGO', 'RISKY', 'REAR', 'LATERAL', 'LEFT0', 'LEFT1', 'OPPOSITE', 'RIGHT0']
        self.motor_list = ['steeringLeft', 'steeringRight', 'motorLeft', 'motorRight']
        self.lidar_list = ['lidar0', 'lidar1', 'lidar2']
        self.collision_list =['front', 'frontleft', 'frontright', 'left', 'right', 'rearleft', 'rearright', 'rear',\
                              'frontall', 'frontleftall', 'frontrightall', 'leftall',\
                              'rightall', 'rearleftall', 'rearrightall', 'rearall']
        
        # Set handles
        self.ego_handle = vrep.simxGetObjectHandle(self.clientID, 'EGO', vrep.simx_opmode_blocking)[1]
        self.risky_handle = vrep.simxGetObjectHandle(self.clientID, 'RISKY', vrep.simx_opmode_blocking)[1]
        self.rear_handle = vrep.simxGetObjectHandle(self.clientID, 'REAR', vrep.simx_opmode_blocking)[1]
        self.lateral_handle = vrep.simxGetObjectHandle(self.clientID, 'LATERAL', vrep.simx_opmode_blocking)[1]
        self.left0_handle = vrep.simxGetObjectHandle(self.clientID, 'LEFT0', vrep.simx_opmode_blocking)[1]
        self.left1_handle = vrep.simxGetObjectHandle(self.clientID, 'LEFT1', vrep.simx_opmode_blocking)[1]
        self.opposite_handle = vrep.simxGetObjectHandle(self.clientID, 'OPPOSITE', vrep.simx_opmode_blocking)[1]
        self.right0_handle = vrep.simxGetObjectHandle(self.clientID, 'RIGHT0', vrep.simx_opmode_blocking)[1]

        for name in self.motor_list:
            self.ego_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'ego_' + name, vrep.simx_opmode_blocking)[1]
            self.risky_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'risky_' + name, vrep.simx_opmode_blocking)[1]
            self.rear_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'rear_' + name, vrep.simx_opmode_blocking)[1]
            self.lateral_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'lateral_' + name, vrep.simx_opmode_blocking)[1]
            self.left0_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'left0_' + name, vrep.simx_opmode_blocking)[1]
            self.left1_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'left1_' + name, vrep.simx_opmode_blocking)[1]
            self.opposite_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'opposite_' + name, vrep.simx_opmode_blocking)[1]
            self.right0_handles[name] = vrep.simxGetObjectHandle(self.clientID, 'right0_' + name, vrep.simx_opmode_blocking)[1]

        for name in self.collision_list:
            self.collision_handles[name] = vrep.simxGetCollisionHandle(self.clientID, name, vrep.simx_opmode_blocking)[1]
        for name in self.lidar_list:
            self.lidar_handles[name] = vrep.simxGetObjectHandle(self.clientID, name, vrep.simx_opmode_blocking)[1]

        # Synchronous refresh
        vrep.simxSynchronous(self.clientID, False)
        vrep.simxSynchronous(self.clientID, True)

        # Discontinue communications for memory clearing
        for name in self.lidar_list:
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_discontinue)
        for name in self.collision_list:
            vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_discontinue)
        vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_discontinue)

        # Set some constants
        self.risky_speed_set = [0, 30, 40, 50, 60]
        self.target = np.array([-1.775, -9.45])

    def get_lidar(self):
        lidar_array = []
        for name in self.lidar_list:
            buffer = vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_buffer)[2]
            lidar_array += buffer
        return lidar_array

    def reset(self):
        # Discontinue communications for memory clearing
        for name in self.collision_list:
            vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_discontinue)
        vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_discontinue)
        for name in self.lidar_list:
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_discontinue)

        # Simulation stop and go
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_oneshot)
        
        # self.risky_speed = self.risky_speed_set[np.random.randint(len(self.risky_speed_set))]

        # Start streaming simulation data
        for name in self.collision_list:
            vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_streaming)
        vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_streaming)
        vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_streaming)
        for name in self.lidar_list:
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_streaming)
        
        # Random choice of scenario
        self.scenario_num = np.random.randint(3)

        self.lidar_sequence = []
        # Rule-based control
        if self.scenario_num == 0:
            # Set risky position
            vrep.simxSetObjectPosition(self.clientID, self.risky_handle, -1, (28.725, 2.275, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.risky_handle, -1, (0.0, -np.pi/2.0, 0.0), vrep.simx_opmode_oneshot)
            # Set rear position
            vrep.simxSetObjectPosition(self.clientID, self.rear_handle, -1, (-3.1, 19.725, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.rear_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set lateral position
            vrep.simxSetObjectPosition(self.clientID, self.lateral_handle, -1, (-6.875, 13.4, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.lateral_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set opposite position
            vrep.simxSetObjectPosition(self.clientID, self.opposite_handle, -1, (2.3498, -12.675, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.opposite_handle, -1, (-np.pi/2.0, 0.0, -np.pi/2.0), vrep.simx_opmode_oneshot)

            if np.random.uniform() < 0.7:
                rearmoving = True
            else:
                rearmoving = False

            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action('ego', [-1.0, 0.0])
            self.set_action('risky', [-1.0, 0.0])
            self.set_action('rear', [-1.0, 0.0])
            self.set_action('opposite', [-1.0, 0.0])
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)

            for i in range(35):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-1.0, 0.0])
                self.set_action('risky', [1.0, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(10):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(15):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                if rearmoving:
                    self.set_action('rear', [-0.5, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(4):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                if rearmoving:
                    self.set_action('rear', [-0.5, 0.0])
                self.lidar_sequence += self.get_lidar()
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
                
            # vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot)

        if self.scenario_num == 1:
            # Set risky position
            vrep.simxSetObjectPosition(self.clientID, self.risky_handle, -1, (-37.0, -1.525, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.risky_handle, -1, (0.0, np.pi/2.0, -np.pi), vrep.simx_opmode_oneshot)
            # Set rear position
            vrep.simxSetObjectPosition(self.clientID, self.rear_handle, -1, (-3.1, 19.725, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.rear_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set lateral position
            vrep.simxSetObjectPosition(self.clientID, self.lateral_handle, -1, (-7.0751, 17.925, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.lateral_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set opposite position
            vrep.simxSetObjectPosition(self.clientID, self.opposite_handle, -1, (2.3498, -12.675, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.opposite_handle, -1, (-np.pi/2.0, 0.0, -np.pi/2.0), vrep.simx_opmode_oneshot)


            if np.random.uniform() < 0.7:
                rearmoving = True
            else:
                rearmoving = False

            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action('ego', [-1.0, 0.0])
            self.set_action('risky', [-1.0, 0.0])
            self.set_action('rear', [-1.0, 0.0])
            self.set_action('opposite', [-1.0, 0.0])
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)
            
            for i in range(20):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-1.0, 0.0])
                self.set_action('risky', [1.0, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(15):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                self.set_action('opposite', [-0.6, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(30):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                if rearmoving:
                    self.set_action('rear', [-0.4, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(4):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [1.0, 0.0])
                if rearmoving:
                    self.set_action('rear', [-0.4, 0.0])
                self.lidar_sequence += self.get_lidar()
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
                
            # vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot)


        if self.scenario_num == 2:
            # Set risky position
            vrep.simxSetObjectPosition(self.clientID, self.risky_handle, -1, (-27.0, -1.95, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.risky_handle, -1, (0.0, np.pi/2.0, -np.pi), vrep.simx_opmode_oneshot)
            # Set rear position
            vrep.simxSetObjectPosition(self.clientID, self.rear_handle, -1, (-3.1, 19.725, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.rear_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set lateral position
            vrep.simxSetObjectPosition(self.clientID, self.lateral_handle, -1, (-7.0751, 17.925, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.lateral_handle, -1, (np.pi/2.0, 0.0, np.pi/2.0), vrep.simx_opmode_oneshot)
            # Set opposite position
            vrep.simxSetObjectPosition(self.clientID, self.opposite_handle, -1, (2.3498, -12.675, 0.3677), vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.opposite_handle, -1, (-np.pi/2.0, 0.0, -np.pi/2.0), vrep.simx_opmode_oneshot)

            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action('ego', [-1.0, 0.0])
            self.set_action('risky', [-1.0, 0.0])
            self.set_action('rear', [-1.0, 0.0])
            self.set_action('opposite', [-1.0, 0.0])
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)

            for i in range(20):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-1.0, 0.0])
                self.set_action('risky', [0.7, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(10):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [0.7, 0.0])
                self.set_action('opposite', [-0.7, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(25):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [0.7, 0.0])
                self.set_action('rear', [-0.4, 0.0])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(3):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [0.5, 0.9])
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)
            for i in range(4):
                vrep.simxPauseCommunication(self.clientID, 1)
                self.set_action('ego', [-0.4, 0.0])
                self.set_action('risky', [0.5, 0.9])
                self.lidar_sequence += self.get_lidar()
                vrep.simxPauseCommunication(self.clientID, 0)
                vrep.simxSynchronousTrigger(self.clientID)

            # for i in range(23):
            #     vrep.simxPauseCommunication(self.clientID, 1)
            #     self.set_action('ego', [-0.4, -1.0])
            #     self.set_action('risky', [0.5, 0.9])
            #     self.lidar_sequence += self.get_lidar()
            #     vrep.simxPauseCommunication(self.clientID, 0)
            #     vrep.simxSynchronousTrigger(self.clientID)
            # for i in range(20):
            #     vrep.simxPauseCommunication(self.clientID, 1)
            #     self.set_action('ego', [-0.4, -1.0])
            #     self.set_action('risky', [0.7, 0.0])
            #     self.lidar_sequence += self.get_lidar()
            #     vrep.simxPauseCommunication(self.clientID, 0)
            #     vrep.simxSynchronousTrigger(self.clientID)
                
            # vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot)
            # keep risky's action for 33 more steps in step()
        
        currentPosition = np.asarray(vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_buffer)[1])
        toTarget = self.target - currentPosition[0:2]
        dist_target = np.sqrt(toTarget[0]**2 + toTarget[1]**2)
        orientation = vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_buffer)[1][0]
        angle = np.arctan2(toTarget[1], toTarget[0])
        targetangle = angle - orientation
        if targetangle > np.pi:
            targetangle -= 2 * np.pi
        elif targetangle < -np.pi:
            targetangle += 2 * np.pi
        targetangle = targetangle / np.pi
        dist_target = dist_target / 20.
        target_state = np.array([targetangle, dist_target])
        # lidar_data = self.get_lidar()

        state = self.lidar_sequence + [-0.4, 0.0]
        # print(self.scenario_num)
        return state

    def step(self, action, timestep):
        done = False

        vrep.simxPauseCommunication(self.clientID, 1)
        self.set_action('ego', action)
        if self.scenario_num == 2:
            if timestep < 23:
                self.set_action('risky', [0.5, 0.9])
            else:
                self.set_action('risky', [0.7, 0.0])
        else:
            self.set_action('risky', [1.0, 0.0])
        vrep.simxPauseCommunication(self.clientID, 0)
        vrep.simxSynchronousTrigger(self.clientID)

        self.lidar_sequence = self.lidar_sequence[96:] + self.get_lidar()
        
        
        
        # Get collision area
        collision = {}
        for name in self.collision_list:
            collision[name] = vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_buffer)[1]

        # Get collision angle
        orientation = 0.0
        while 1:
            if vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_buffer)[0] == vrep.simx_return_ok:
                orientation = vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_buffer)[1][0] * 180.0 / np.pi
                break
        if 90.0 <= orientation < 180.0:
            orientation = 180.0 - orientation
        elif -180.0 <= orientation < -90.0:
            orientation += 180.0
        elif -90.0 <= orientation < 0:
            orientation = -orientation
        
        # Get alpha and beta
        if collision['front'] or collision['frontright'] or collision['frontleft'] or collision['right']\
        or collision['rearright'] or collision['rear'] or collision['rearleft'] or collision['left']:
            done = True
            if collision['front'] or collision['frontright'] or collision['frontleft']:
                if collision['front'] or collision['frontleft']:
                    alpha = 2.698
                elif collision['frontright']:
                    alpha = 2.038

                if 15.0 <= orientation < 45.0:
                    beta = 2.868
                elif 45.0 <= orientation < 75.0:
                    beta = 2.662
                elif 0.0 <= orientation < 15.0:
                    beta = 2.772
                else:
                    beta = 2.662

            if collision['rearright'] or collision['rear'] or collision['rearleft']:
                alpha = 1.991
                beta = 2.780

            if collision['right']:
                alpha = 2.038
                if 15.0 <= orientation < 45.0:
                    beta = 2.504
                elif 45.0 <= orientation < 75.0:
                    beta = 2.401
                elif 75.0 <= orientation <= 90.0:
                    beta = 2.429
                elif 0 <= orientation < 15.0:
                    beta = 2.881

            if collision['left']:
                alpha = 2.698
                if 15.0 <= orientation < 45.0:
                    beta = 2.899
                elif 45.0 <= orientation < 75.0:
                    beta = 2.433
                elif 75.0 <= orientation <= 90.0:
                    beta = 2.765
                else:
                    beta = 2.899
            
            # Crashing reward
            reward = -np.exp(2.011*10**(-7)*0.5*1500*(np.exp(81.95*0.008+beta+1.2052*0.368))**2+alpha)
        elif collision['frontall'] or collision['frontrightall'] or collision['frontleftall'] or collision['rightall'] \
            or collision['rearrightall'] or collision['rearall'] or collision['rearleftall'] or collision['leftall']:
            reward = -5.
            done = True
        elif timestep > 40:
            done = True
            reward = 0.
        else:
            reward = 0

        currentPosition = np.asarray(vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_buffer)[1])
        toTarget = self.target - currentPosition[0:2]
        dist_target = np.sqrt(toTarget[0]**2 + toTarget[1]**2)
        orientation = vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_buffer)[1][0]
        angle = np.arctan2(toTarget[1], toTarget[0])
        targetangle = angle - orientation
        if targetangle > np.pi:
            targetangle -= 2 * np.pi
        elif targetangle < -np.pi:
            targetangle += 2 * np.pi
        targetangle = targetangle / np.pi
        dist_target = dist_target / 20.
        target_state = np.array([targetangle, dist_target])
        state1 = self.lidar_sequence + action

        if done == True:
            vrep.simxPauseCommunication(self.clientID, 1)
            self.set_action('ego', [-1.0, 0.0])
            self.set_action('risky', [-1.0, 0.0])
            self.set_action('rear', [-1.0, 0.0])
            self.set_action('opposite', [-1.0, 0.0])
            vrep.simxPauseCommunication(self.clientID, 0)
            vrep.simxSynchronousTrigger(self.clientID)

        # Update prev_distance
        self.prev_distance = dist_target

        return state1, reward, done

    # Send action signal to simulator
    def set_action(self, type, action):
        d = 0.755
        l = 2.5772
        desiredSpeed, desiredSteeringAngle = action
        desiredSpeed = 25. * desiredSpeed + 25
        desiredSteeringAngle = 0.785 * desiredSteeringAngle

        if desiredSteeringAngle != 0.:
            steeringAngleLeft = np.arctan(l / (-d + l / np.tan(desiredSteeringAngle)))
            steeringAngleRight = np.arctan(l / (d + l / np.tan(desiredSteeringAngle)))
        else:
            steeringAngleLeft = 0.
            steeringAngleRight = 0.
        
        if type == 'ego':
            vrep.simxSetJointTargetVelocity(self.clientID, self.ego_handles['motorLeft'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.clientID, self.ego_handles['motorRight'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.ego_handles['steeringLeft'], steeringAngleLeft, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.ego_handles['steeringRight'], steeringAngleRight, vrep.simx_opmode_oneshot)
        elif type == 'risky':
            vrep.simxSetJointTargetVelocity(self.clientID, self.risky_handles['motorLeft'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.clientID, self.risky_handles['motorRight'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.risky_handles['steeringLeft'], steeringAngleLeft, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.risky_handles['steeringRight'], steeringAngleRight, vrep.simx_opmode_oneshot)
        elif type == 'rear':
            vrep.simxSetJointTargetVelocity(self.clientID, self.rear_handles['motorLeft'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.clientID, self.rear_handles['motorRight'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.rear_handles['steeringLeft'], steeringAngleLeft, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.rear_handles['steeringRight'], steeringAngleRight, vrep.simx_opmode_oneshot)
        elif type == 'opposite':
            vrep.simxSetJointTargetVelocity(self.clientID, self.opposite_handles['motorLeft'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(self.clientID, self.opposite_handles['motorRight'], desiredSpeed, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.opposite_handles['steeringLeft'], steeringAngleLeft, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetPosition(self.clientID, self.opposite_handles['steeringRight'], steeringAngleRight, vrep.simx_opmode_oneshot)

    def close(self):
        for name in self.collision_list:
            vrep.simxReadCollision(self.clientID, self.collision_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxGetObjectOrientation(self.clientID, self.ego_handle, self.risky_handle, vrep.simx_opmode_discontinue)
        vrep.simxGetObjectPosition(self.clientID, self.ego_handle, -1, vrep.simx_opmode_discontinue)
        for name in self.lidar_list:
            vrep.simxGetVisionSensorDepthBuffer(self.clientID, self.lidar_handles[name], vrep.simx_opmode_discontinue)
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)
        time.sleep(0.5)
        vrep.simxSynchronous(self.clientID, False)
        