
import numpy as np
# lander variables
UNLOADED_MASS = 100.0 #KG
FUEL_MASS = 100.0 #KG
FUEL_RATE = 0.5 # KGS-1
GRAVITY = 6.673e-11 
MARS_MASS = 6.42e23 #kg
DRAG_COEF_LANDER = 1.0
LANDER_SIZE = 1.0
DRAG_COEF_CHUTE = 2.0
MARS_RADIUS = 3386000.0
MAX_THRUST = 1.5*(UNLOADED_MASS+FUEL_MASS)*GRAVITY*MARS_MASS/(MARS_RADIUS**2)
EXOSPHERE = 200_000.0
delta_t =0.1



def atmospheric_density(x):
    alti = x - MARS_RADIUS
    if alti > EXOSPHERE or alti < 0.0:
        return 0.0
    else:
        return 0.017*np.exp(-alti/11000.0)


class lander:
    def __init__(self, init_altitude, velocity =0 ):
        self.xprev = None
        self.xnew = None
        self.x = init_altitude + MARS_RADIUS
        self.v = velocity
        self.mass = UNLOADED_MASS + FUEL_MASS
        self.a = None
        self.parachute_deployed = False
       
    def move(self, delta_t, step, throttle=0, para_deploy =False):
        atmos_den = atmospheric_density(self.x)
        gravity = -(GRAVITY*MARS_MASS*(self.mass))/((self.x)**2)
        dragv = -0.5*(atmos_den)*DRAG_COEF_LANDER*np.pi*(LANDER_SIZE**2)*(self.v)**2*np.sign(self.v)
        dragp = -0.5*(atmos_den)*DRAG_COEF_CHUTE*5*((2*LANDER_SIZE)**2)*(self.v)**2*np.sign(self.v)

        if self.mass > UNLOADED_MASS:
            thrust = throttle*MAX_THRUST
        else:
            thrust = 0

        if para_deploy:
            self.a  = (gravity+dragp+dragv+thrust)/ self.mass
        else:
            self.a = (gravity+dragv+thrust)/self.mass


        if step == 0:
            self.xnew = self.x + self.v*delta_t
            self.v += self.a*delta_t
        else:
            self.xnew = 2*self.x - self.xprev + self.a*delta_t**2
            self.v = (1/delta_t)*(self.xnew-self.x)


        self.xprev = self.x
        self.x = self.xnew

        if self.mass > UNLOADED_MASS:
            self.mass -= throttle*FUEL_RATE*delta_t
        else:
            self.mass = UNLOADED_MASS

    def action(self, throttle, stepi):
        self.move(delta_t, stepi, throttle )

class landerEnv:

    def reset(self, alt):
        self.a_max = 0
        self.landerX = lander(alt)
        obs = [self.landerX.x - MARS_RADIUS, self.landerX.v]
        return obs
    
    def step(self, action, step):
        self.landerX.action(action, step)
        self.a_max = max(self.a_max, abs(self.landerX.a))

        new_obeservation = [self.landerX.x - MARS_RADIUS, self.landerX.v]

        if self.landerX.x < MARS_RADIUS:
            reward = -(abs(self.landerX.v))
        else:
            reward = 0

        if self.landerX.x < MARS_RADIUS:
            done = True
        else:
            done = False
        return new_obeservation, reward, done